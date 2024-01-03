# This is pretty similar to cbpdn_paramscan.py, but re-importing messes with the GPU memory, so we need to do it here
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm

import cbpdn_paramscan as scan
from sporco.dictlrn.onlinecdl import OnlineConvBPDNDictLearn as DictLearn
import torch 
import torch_cbpdn
import sporco.metric as sm

from sporco import util
from sporco import signal
from sporco import plot

from SLIP import Image


# This is to push everything to the GPU
def new_xstep(self, S, lmbda, dimK) :
    torch.cuda.empty_cache()
    self.device = torch.device('cuda:0')
    
    S_torch = torch.tensor(S, dtype = torch.float32, device = self.device)   
    D_torch = torch.tensor(self.D, dtype = torch.float32, device = self.device) 
    solver = torch_cbpdn.CBPDN(D = D_torch, S = S_torch, device = self.device , lmbda = lmbda,
                            MaxMainIter = self.opt['CBPDN']['MaxMainIter'], RelStopTol = self.opt['CBPDN']['RelStopTol'],
                            RelaxParam = self.opt['CBPDN']['RelaxParam'], L1Weight = self.opt['CBPDN']['L1Weight'],
                            AutoRho_RsdlRatio = self.opt['CBPDN']['AutoRho']['RsdlRatio'],
                            dimK=dimK, dimN=self.cri.dimN)
    
    solver.solve() 
    self.primal = sm.psnr(S, solver.reconstruct().cpu().numpy())
    self.dual = 1 - (np.count_nonzero(solver.X.cpu().numpy()) / solver.X.cpu().numpy().size)
    
    torch.cuda.empty_cache()
    
    self.Sf = solver.Sf.unsqueeze(-1).cpu().numpy()
    self.setcoef(solver.X.cpu().numpy())
    self.xstep_itstat = None

# We also need to update the iteration stats to return something we can actually use
def zpad(v, Nv):
    vp = np.zeros(Nv + v.shape[len(Nv):], dtype=v.dtype)
    axnslc = tuple([slice(0, x) for x in v.shape])
    vp[axnslc] = v
    return vp
def new_iteration_stats(self):
    """Construct iteration stats record tuple."""

    tk = self.timer.elapsed(self.opt['IterTimer'])
    if self.xstep_itstat is None:
        objfn = (0.0,) * 3
        rsdl = (self.primal,
                self.dual) # todo here compute primalrsdl and dualrsdl
        rho = (0.0,)
    else:
        objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                    self.xstep_itstat.RegL1)
        rsdl = (self.xstep_itstat.PrimalRsdl,
                self.xstep_itstat.DualRsdl)
        rho = (self.xstep_itstat.Rho,)

    cnstr = np.linalg.norm(zpad(self.D, self.cri.Nv) - self.G)
    dltd = np.linalg.norm(self.D - self.Dprv)

    tpl = (self.j,) + objfn + rsdl + rho + (cnstr, dltd, self.eta) + \
            self.itstat_extra() + (tk,)
    return type(self).IterationStats(*tpl)


# And one to load a source database, in case its not already been processed by the rests of the scirpts 
def load_S(database_path, DEZOOM): 

    img_db = util.ExampleImages(pth=database_path)
    img = img_db.image(img_db.images()[0], zoom=DEZOOM,  scaled=True, gray=True)
    N_X, N_Y = img.shape
    print(N_X, N_Y)

    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    im = Image(pe=parameterfile)
    im.set_size((N_X, N_Y))

    # a stack
    N_image = len(img_db.images())
    S = np.zeros((img.shape[0], img.shape[1], N_image))

    list_images = img_db.images()
    #np.random.shuffle(list_images)
    for i_image, fname in tqdm(enumerate(img_db.images()[:N_image]), total = N_image, desc = 'Preprocessing database into SPORCO format'):
        img = img_db.image(fname, zoom=DEZOOM, scaled=True, gray=True)
        white = im.whitening(img)
        white, _, _ = signal.local_contrast_normalise(white)
        white *= im.mask
        S[:, :, i_image] = white

    return S

# And one to load a source database, in case its not already been processed by the rests of the scirpts 
def learn(database_path, DEZOOM=1,
        init_dico=None, N_image = 250,
        lmbda=0.01, n_epochs=1000, max_iter=100, N_image_rec=250,
        eta_a=100.0, eta_b=1000.0, AutoRho=False, RsdlRatio=1.05, Period=1., Scaling=1.,
        RelStopTol=1e-3, rho=10.0, RelaxParam=0.8, AuxVarObj=False,
        seed=42):

    # make a seeded choice for the testing images 
    np.random.seed(seed)
    
    S = load_S(database_path=database_path, DEZOOM=DEZOOM)
    N_image = S.shape[-1]
    
    learning_images_idxs = np.random.randint(0, N_image, n_epochs)
    images_idxs = np.random.randint(0, N_image, N_image_rec)
    

    # Set regularization parameter and options for dictionary learning solver.
    opt = DictLearn.Options({
                    'Verbose':False,
                    'eta_a':eta_a, 'eta_b':eta_b, 'DataType':np.float32,
                    'CBPDN':{'rho': rho, 
                            'AutoRho': {'Enabled': AutoRho, 'Period': Period, 'Scaling': Scaling, 'RsdlRatio': RsdlRatio},
                            'RelaxParam': RelaxParam, 'RelStopTol': RelStopTol, 'MaxMainIter': max_iter,
                            'FastSolve': False, 'DataType': np.float32, 'AuxVarObj': AuxVarObj},
                    })

    # Create solver object and solve.
    d = DictLearn(init_dico, lmbda, opt)

    for it in tqdm(range(n_epochs), total = n_epochs, desc = 'Learning . . .'):
        i_image = learning_images_idxs[it]
        d.solve(S[:, :, i_image])
        
    rec = np.zeros(N_image_rec)
    cost = np.zeros(N_image_rec)
    for it_test in tqdm(range(N_image_rec), total = N_image_rec, desc = 'Testing . . .'):
        i_image_test = images_idxs[it_test]
        d.solve(S[:, :, i_image_test]) # pick and solve for one random image
        rec[it_test] = d.itstat[-1].PrimalRsdl
        cost[it_test] = d.itstat[-1].DualRsdl

    return cost, rec, d

def learn_from_img(database_path, D, save_name, n_epochs, N_image_rec, DEZOOM=1) :
    # First we change the functions to a PyTorch faster version
    #DictLearn.iteration_stats = scan.new_iteration_stats # black magic
    #DictLearn.xstep = scan.new_xstep # black magic 
    
    init_dico = D.copy()
    cost, rec, d = learn(database_path = database_path, DEZOOM = 1,
                        init_dico=init_dico, n_epochs=n_epochs, N_image_rec=N_image_rec,
                        AuxVarObj = False, max_iter = 300, lmbda = 0.01, rho = 10.0, 
                        AutoRho = False, RelStopTol = 1e-2, Period = 1, Scaling = 1, RsdlRatio = 1.05,
                        eta_a = 100, eta_b = 1000, RelaxParam = 1.8,)
    learned_dict = d.getdict().squeeze()
    np.savez_compressed(save_name, rec=rec, D=learned_dict)
    
    fig = plot.figure(figsize=(10, 10))
    plot.subplot(1, 1, 1)
    plot.imview(util.tiledict(learned_dict), fig=fig)
    plt.title('Learned dict, saved to %s ' % save_name)
    fig.show()
    