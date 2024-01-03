import numpy as np 
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
import time 

from sporco.dictlrn.onlinecdl import OnlineConvBPDNDictLearn as DictLearn
from sporco import util
from sporco import signal
import torch 
import torch_cbpdn
import sporco.metric as sm

# ----------------- #
# --- Replacement-- #
# ----------------- #
# We start by replacing the X step of the OnlineConvBPDNDictLearn class
# This is so we can boost it up in PyTorch, but still use the rest of the class that is nigh impossible to translate in PyTorch
def new_xstep(self, S, lmbda, dimK) :
    torch.cuda.empty_cache()
    self.device = torch.device('cuda:0')
    
    S_torch = torch.tensor(S, dtype = torch.float64, device = self.device)   
    D_torch = torch.tensor(self.D, dtype = torch.float64, device = self.device) 
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


# ----------------- #
# --- Functions --- #
# ----------------- #
# Now some functions for the parameter scan 
# to create empty files for parallel processing
def touch(fname): open(fname, 'w').close()

def dico_plot(dico, ncols, nrows,
              vmin=-1, vmax=None, cmap=plt.cm.gray,
              wspace=.5, hspace=.5):
    
    fig, axs = plt.subplots(figsize=(10, 10*nrows/ncols),
                            ncols=ncols, nrows=nrows)

    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if vmax is None:
            vmax = np.max(dico[:, :, i])
            vmin = - vmax
        ax.imshow(dico[:, :, i], interpolation = 'none', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('off')

    plt.subplots_adjust(wspace=wspace/ncols, hspace=hspace/nrows)
    return fig, axs

# And one to load a source database, in case its not already been processed by the rests of the scirpts 
def load_S(database_path, DEZOOM, do_contrast_normalization, do_highpass): 

    img_db = util.ExampleImages(pth=database_path)
    img = img_db.image(img_db.images()[0], zoom=1/DEZOOM,  scaled=True, gray=True)
    N_X, N_Y = img.shape

    # load SLIP and set image size
    from SLIP import Image
    parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'
    im = Image(pe=parameterfile)
    im.set_size((N_X, N_Y))

    # a stack
    N_image = len(img_db.images())
    S = np.zeros((img.shape[0], img.shape[1], N_image))

    list_images = img_db.images()
    np.random.shuffle(list_images)
    for i_image, fname in tqdm(enumerate(img_db.images()[:N_image]), total = N_image, desc = 'Preprocessing database into SPORCO format'):
        img = img_db.image(fname, zoom=1/2, scaled=True, gray=True)
        white = im.whitening(img)
        if do_contrast_normalization:
            white, _, _ = signal.local_contrast_normalise(white)
        if do_highpass:
            sl, white = signal.tikhonov_filter(white, 10, 16)
        white *= im.mask
        S[:, :, i_image] = white

    return S

# And one to load a source database, in case its not already been processed by the rests of the scirpts 
def learn(database_path, DEZOOM=2, do_contrast_normalization=False, do_highpass=True,
        init_dico=None, filter_size=12, K=12*8*2, N_image = 250,
        lmbda=0.01, n_epochs=1000, max_iter=100, N_image_rec=50,
        eta_a=100.0, eta_b=1000.0, AutoRho=False, RsdlRatio=1.05, Period=1., Scaling=1.,
        RelStopTol=1e-3, ZeroMean=True, do_mask=True, rho=10.0, RelaxParam=0.8, AuxVarObj=False,
        seed=42, verbose=True):

    S = load_S(database_path=database_path, DEZOOM=DEZOOM, do_contrast_normalization=do_contrast_normalization, do_highpass=do_highpass)

    if init_dico is None:
        init_dico = np.random.randn(filter_size, filter_size, K)
    if do_mask:
        coords = np.linspace(-1, 1, num=filter_size+2)[1:-1]
        x, y = np.meshgrid(coords, coords, indexing='ij')
        r = np.sqrt(x **2 + y **2)
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L220
        mask_exponent = 4.
        filter_mask_2D = ((np.cos(np.pi*r)+1)/2 * (r < 1.))**(1./mask_exponent)
        init_dico *= filter_mask_2D[:, :, None]

    # Set regularization parameter and options for dictionary learning solver.
    opt = DictLearn.Options({
                    'Verbose':False, #'ZeroMean':ZeroMean, 
                    'eta_a':eta_a, 'eta_b':eta_b, 'DataType':np.float64,
                    'CBPDN':{'rho': rho, 
                            'AutoRho': {'Enabled': AutoRho, 'Period': Period, 'Scaling': Scaling, 'RsdlRatio': RsdlRatio},
                            'RelaxParam': RelaxParam, 'RelStopTol': RelStopTol, 'MaxMainIter': max_iter,
                            'FastSolve': False, 'DataType': np.float64, 'AuxVarObj': AuxVarObj},
                    # 'CCMOD': {'rho': rho, 'ZeroMean':ZeroMean},
                    # dmethod:'cns',
                    })

    # Create solver object and solve.
    d = DictLearn(init_dico, lmbda, opt)# , dmethod='cns')

    for it in tqdm(range(n_epochs), total = n_epochs, desc = 'Learning . . .'):
        i_image = np.random.randint(0, N_image)
        d.solve(S[:, :, i_image])
        
    rec = np.zeros(N_image_rec)
    cost = np.zeros(N_image_rec)
    for i_image in tqdm(range(N_image_rec), total = N_image_rec, desc = 'Testing . . .'):
        C = d.solve(S[:, :, np.random.randint(0, N_image)]) # pick and solve for one random image
        rec[i_image] = d.itstat[-1].PrimalRsdl
        cost[i_image] = d.itstat[-1].DualRsdl

    return cost, rec, d


# ----------------- #
# --- Params    --- #
# ----------------- #
def run_scan(database_path, D, reload_only = False) :
    print('>>Running parameter scan for learning the dictionnaries<<')
    
    # And now we update the classes
    DictLearn.iteration_stats = new_iteration_stats # black magic
    #DictLearn.xstep = new_xstep # black magic       

    # Now for some general parameters 
    DEBUG = 2 # Granularity of the runs
    n_epochs_default = 25//DEBUG # boost to 5000

    filter_size = 12 # size of each dico's element
    K = 12 * 8 * 2

    # Figure parameters ----------------------------------------------------
    fig_width = 10

    N_scan = 8 # move to 8
    scan_line = np.logspace(-1, 1, N_scan, base=10, endpoint=True)
    scan_line_small = np.logspace(-1, 1, N_scan//2, base=10, endpoint=True)

    scan_dict = {'lmbda': 0.01 * scan_line,
                #'eta_a': 100.0 * scan_line,
                #'eta_b': 1000.0 * scan_line,
                #'rho': 10.0 * scan_line,
                'max_iter': [100//10, 100//3, 100, 100*3, 100*10],
                #  'n_epochs': [n_epochs//2, n_epochs, n_epochs*2],
                #'Period': 1. * scan_line_small,
                #'Scaling': 1.0 * scan_line_small,
                #'RsdlRatio': 1.05 * scan_line,
                'RelaxParam': np.linspace(0, 2, N_scan+2 , endpoint=True)[1:-1], # between 0 and 2
                #'RelStopTol': 1e-3 * scan_line_small,
                #'do_mask': [not(True)],
                #'ZeroMean': [not(True)],
                #'AuxVarObj': [not(False)],
                #'AutoRho': [not(False)],
                #'AuxVarObj': [not(False)],
                'filter_size': [5, 8, 13, 21], 
                #'K': [89, 144, 233, 377, 610, 987, 1364, 2351],
                }

    init_dico = D.copy()
    rec_dict = {}
    cost_dict = {}
    
    if not reload_only :
        tic = time.time()
            
        '''# Default values, just for the sake of it
        cost, rec, d = learn(database_path = database_path, DEZOOM = 2, do_contrast_normalization = True, do_highpass = False, 
                            verbose=True, init_dico=init_dico, n_epochs=n_epochs_default,
                            do_mask = True, ZeroMean = True, AuxVarObj = False, max_iter = 100, lmbda = 0.01, rho = 10.0, 
                            AutoRho = False, RelStopTol = 1e-3, Period = 1, Scaling = 1, RsdlRatio = 1.05,
                            eta_a = 100, eta_b = 1000, RelaxParam = 1.8, filter_size = filter_size, K = K)
        D = d.getdict()
        D = D.squeeze()
        cputime = time.time() - tic
        print(f'For default values, Reconstruction={rec.mean():.3e}, Cost={cost.mean():.3e} - done in {cputime:.1f} s')'''
        #fig, axs = dico_plot(D, ncols=13, nrows=5)
        #plt.show(block = False)


        # Now this is where the fun begins
        N_values = 0
        for variable in scan_dict.keys():
            N_values += len(scan_dict[variable])

        i_values = 0
        for variable in scan_dict.keys():
            print(20*'‼️', variable, 20*'‼️')
            
            N_value = len(scan_dict[variable])
            for i_value, value in enumerate(scan_dict[variable]):
                i_values += 1
                if type(value) in [bool, int, str]:
                    scan_tag = f'{variable}={value}'
                else:
                    scan_tag = f'{variable}={value:.3e}'

                kwargs = {variable: value}
                if variable in ['K', 'filter_size']:
                    init_dico_ = None
                else:
                    init_dico_ = init_dico.copy()

                tic = time.time()
                cost, rec, d = learn(database_path = database_path, verbose=True, n_epochs=n_epochs_default, init_dico=init_dico_, **kwargs)
                D = d.getdict()
                D = D.squeeze()
                cputime = time.time() - tic

                rec_dict[scan_tag] = rec.mean()
                cost_dict[scan_tag] = cost.mean()
                
                print(f'For {scan_tag} ({i_value+1}/{N_value} - {i_values}/{N_values}), Reconstruction={rec.mean():.3e}, Cost={cost.mean():.3e} - done in {cputime:.1f} s')
                fig, axs = dico_plot(D, ncols=13, nrows=5)
                plt.show(block = False)
                fig.savefig(f'./scans/fig_paramscans_{scan_tag}.pdf', bbox_inches = 'tight', transparent = True, dpi = 200)
                plt.close(fig)
                print(50*'‼️')

        # Save the results
        np.save('./data/scan_rect_dict.npy', rec_dict)
        np.save('./data/scan_cost_dict.npy', cost_dict)
                        

    if reload_only :
        rec_dict = np.load('./data/scan_rect_dict.npy', allow_pickle=True).item()
        cost_dict = np.load('./data/scan_cost_dict.npy', allow_pickle=True).item()   
        
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))

    plot_vars = ['lmbda',  'max_iter', 'RelaxParam', 'filter_size',]
    cmaps = [plt.cm.viridis, plt.cm.magma, plt.cm.inferno, plt.cm.plasma]
    for ivar, variable in enumerate(plot_vars):
        recs, costs = [], []
        values = scan_dict[variable]
        
        # Normalize the scan values to [0, 1]
        norm = plt.Normalize(min(values), max(values))
        
        for i_value, value in enumerate(values):
            if type(value) in [bool, int, str]:
                scan_tag = f'{variable}={value}'
            else:
                scan_tag = f'{variable}={value:.3e}'

            # Access rec and cost values from the dictionaries
            rec = rec_dict[scan_tag]
            cost = cost_dict[scan_tag]

            # Get color from colormap based on normalized value
            color = cmaps[ivar](norm(value))
            
            ax.plot([rec], 1-np.asarray([cost]), '.', ms=5, color=color)
            recs.append(rec)
            costs.append(cost)
            
        # Plot the line with a gradient of colors
        for i in range(len(recs) - 1):
            ax.plot(recs[i:i+2], 1-np.asarray(costs[i:i+2]), color=plt.cm.viridis(norm(values[i])), label=f"{variable}={values[i]:.3e}" if i == 0 else "")
        
    ax.set_xlabel('PSNR', fontsize = 16)
    #ax.set_xlim(25, 65)
    #ax.set_ylim(0.992, 1.)
    ax.set_ylabel('sparseness', fontsize = 16)

    #ax.set_xticks ([25, 35, 45, 55, 65])
    #ax.set_yticks ([1, 1-.02, 1-.04, 1-.06, 1-.08])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize = 14)

    ax.legend()

    fig.savefig('./figs/fig_paramscans.pdf', bbox_inches = 'tight', transparent = True, dpi = 200)
    plt.show(block = False)