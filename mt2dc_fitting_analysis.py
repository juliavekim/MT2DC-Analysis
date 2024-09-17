###############################################
# Mt2dc fitting analysis
#The fitting analysis code, which fits two pre-defined functions to the drop-off windows of 21 normalised $\text{m}_{\text{T2}}$($W|\alpha,$ pT$_c$ = FALSE)' histograms, each of fixed $\alpha.$ Various data, including the fitted parameters, number of degrees of freedom, chi-square and derivatives, are saved to the first .txt file. The first fitted parameter in both fits 1 (f1) and 2 (f2), as well as their errors, are respectively saved to the last two .txt files. They shall be later used for the study of which $\alpha$ gives the best mass measurement.
##############################################

import ROOT
import numpy as np 
from array import array 

##############################################
# Define the input and output root files
##############################################
f_inputRoot = ROOT.TFile.Open("/Users/juliakim/Documents/TH2F_plotter_output.root", "read")

parameterFile = open("/Users/juliakim/Documents/styledPlotsOutputs/fit_functions_parameters.txt",'w')
parameterFile.write('alpha, pars 0, 1, 2, 3, 4, chi-squared, NDF, derivative at par0 \n')
parameterFile2 = open("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f1_Parameter0.txt",'w')
parameterFile2.write("mT2prime_W_UC_f1_Parameter0: value, ucty [GeV] \n") 
parameterFile3 = open("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f2_Parameter0.txt",'w')
parameterFile3.write("mT2prime_W_UC_f2_Parameter0: value, ucty [GeV] \n") 
parameterFile4 = open("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f3_Parameter0.txt",'w')
parameterFile4.write("mT2prime_W_UC_f3_Parameter0: value, ucty [GeV] \n")

outDir = "/Users/juliakim/Documents/styledPlotsOutputs/fit_functions/" 
f_outputRoot = ROOT.TFile.Open("/Users/juliakim/Documents/mt2dc_fitting_analysis_output.root", "recreate")
tree = ROOT.TTree("results", "tree storing results") 

# create float array pointers 
alpha = array('d', [0]) 
par0 = array('d', [0])
par0Error = array('d', [0])
par1 = array('d', [0]) 
par2 = array('d', [0]) 
par3 = array('d', [0]) 
par4 = array('d', [0]) 
chisquare = array('d', [0]) 
NDF = array('d', [0]) # number of degrees of freedom 
slope_par0 = array('d', [0]) # derivative at parameter 0 
functionType = array('d', [0]) # f1 (1) or f2 (2) 

# create branches
tree.Branch("alpha", alpha, 'alpha/D') 
tree.Branch("par0", par0, 'par0/D')
tree.Branch("par0Error", par0Error, 'par0Error/D')
tree.Branch("par1", par1, 'par1/D') 
tree.Branch("par2", par2, 'par2/D')
tree.Branch("par3", par3, 'par3/D')
tree.Branch("par4", par4, 'par4/D') 
tree.Branch("chisquare", chisquare, 'chisquare/D') 
tree.Branch("NDF", NDF, 'NDF/D') 
tree.Branch("slope_par0", slope_par0, 'slope_par0/D') 
tree.Branch("functionType", functionType, 'functionType/D') 

##############################################
# Define constants and optimisation mode 
##############################################
m_W = 80 # GeV 
m_t = 173 # GeV
alphaList = np.linspace(0, 1, 21)  

##############################################
# Get histograms for fitting 
##############################################
#ROOT.gStyle.SetOptStat(1111)
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas()

#### Get histograms
h2_mT2prime_W_UC = f_inputRoot.Get("h2_mT2prime_W_UC").Clone() 

#### Get number of alpha bins 
h_proj_x = h2_mT2prime_W_UC.ProjectionX("h_proj_x", 0, -1)
num_alpha_bins = h_proj_x.GetNbinsX()

#### Loop over every alpha bin to create num_alpha_bins TH1 histograms per TH2 histogram
for i in range(1, num_alpha_bins+1):
    print('running iteration', i) 
    h_mT2prime_W_UC_i = h2_mT2prime_W_UC.ProjectionY("h_mT2prime_W_UC_i", i, i) 
    # recall from ROOT file, that mT2prime values are constrained to [40, 300] range, so spike at 0 is omitted  
    
    # Normalise histograms
    h_mT2prime_W_UC_i.Scale(1./h_mT2prime_W_UC_i.Integral(), "width") 
       
    f1 = ROOT.TF1("f1", "([2]+[3]*x)*atan((x-[0])/[1]) + [4]*x + 1", 70, 100) # optimised to alpha = 0.1 
    f1.SetParameters(80, 95, -0.05, 0.0124, -0.0124) 
    
    #f2 = ROOT.TF1("f2", "([2] + [3]*x)*(x-[0])/sqrt([1]+(x-[0])**2) + [4]*x + 1", 70, 100)
    #f2.SetParameters(80, 95, 0.05, 0.0124, -0.0124) #?? 
    
    f2 = ROOT.TF1("f2", "([2]+[3]*x)*atan((x-[0])/[1]) + [4]*x + 1", 70, 90) # optimised to alpha = 1
    f2.SetParameters(80, 120, -0.05, 0.0113, -0.0113)
    
    f3 = ROOT.TF1("f3", "([2]+[3]*x)*atan((x-[0])/[1]) + [4]*x + 1", 70, 100) # optimised to alpha = 0.5 
    f3.SetParameters(80, 100, -0.05, 0.0120, -0.0120) 
                  
    ## mT2(W|alpha)' 
    # f1 
    h_mT2prime_W_UC_i.Fit("f1", "0", "", 70, 100) 
    h_mT2prime_W_UC_i.Draw("E") 
    f1.Draw("same")  
    h_mT2prime_W_UC_i.SetNameTitle("h_mT2prime_W_UC_i", "h_mT2prime_W_UC_" + str(i) + "_f1")
    h_mT2prime_W_UC_i.Write() 
    c.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f1.pdf") 
    #c2.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f1_LOG.pdf") # logarithmic version of plot 
    #h_mT2prime_W_UC_i.SetAxisRange(50, 120, "X") 
    #c3.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f1_LIMITED.pdf") # limited x-axis version of plot 
    h_mT2prime_W_UC_i_f1_data = np.around(np.array([alphaList[i-1], f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2), 
                                                    f1.GetParameter(3), f1.GetParameter(4), f1.GetChisquare(), f1.GetNDF(), 
                                                    f1.Derivative(f1.GetParameter(0))]), 3) 
    parameterFile.write('h_mT2prime_W_UC_' + str(i) + '_f1: \t' + str(h_mT2prime_W_UC_i_f1_data) + '\n')
    parameterFile2.write(str(alphaList[i-1]) + ',' + str(f1.GetParameter(0)) + ',' + str(f1.GetParError(0)) + '\n') 

    alpha[0] = alphaList[i-1] 
    par0[0] = f1.GetParameter(0)
    par1[0] = f1.GetParameter(1)
    par2[0] = f1.GetParameter(2) 
    par3[0] = f1.GetParameter(3)  
    par4[0] = f1.GetParameter(4) 
    chisquare[0] = f1.GetChisquare()
    NDF[0] = f1.GetNDF() 
    slope_par0[0] = f1.Derivative(f1.GetParameter(0))
    functionType[0] = 1 
    tree.Fill()
   
    # f2
    h_mT2prime_W_UC_i.Fit("f2", "0", "", 70, 100) 
    h_mT2prime_W_UC_i.Draw("E") 
    f2.Draw("same")  
    h_mT2prime_W_UC_i.SetNameTitle("h_mT2prime_W_UC_i", "h_mT2prime_W_UC_" + str(i) + "_f2")
    h_mT2prime_W_UC_i.Write() 
    c.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f2.pdf") 
    #c2.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f2_LOG.pdf")
    #h_mT2prime_W_UC_i.SetAxisRange(50, 120, "X") 
    #c3.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f2_LIMITED.pdf") 
    h_mT2prime_W_UC_i_f2_data = np.around(np.array([alphaList[i-1], f2.GetParameter(0), f2.GetParameter(1), f2.GetParameter(2), 
                                                    f2.GetParameter(3), f2.GetParameter(4), f2.GetChisquare(), f2.GetNDF(), 
                                                    f2.Derivative(f2.GetParameter(0))]), 3) 
    parameterFile.write('h_mT2prime_W_UC_' + str(i) + '_f2: \t' + str(h_mT2prime_W_UC_i_f2_data) + '\n') 
    parameterFile3.write(str(alphaList[i-1]) + ',' + str(f2.GetParameter(0)) + ',' + str(f2.GetParError(0)) + '\n') 

    alpha[0] = alphaList[i-1] 
    par0[0] = f2.GetParameter(0)
    par1[0] = f2.GetParameter(1)
    par2[0] = f2.GetParameter(2) 
    par3[0] = f2.GetParameter(3)  
    par4[0] = f2.GetParameter(4) 
    chisquare[0] = f2.GetChisquare()
    NDF[0] = f2.GetNDF() 
    slope_par0[0] = f2.Derivative(f2.GetParameter(0))
    functionType[0] = 1 
    tree.Fill() 

    # f3 
    h_mT2prime_W_UC_i.Fit("f3", "0", "", 70, 100) 
    h_mT2prime_W_UC_i.Draw("E") 
    f3.Draw("same")  
    h_mT2prime_W_UC_i.SetNameTitle("h_mT2prime_W_UC_i", "h_mT2prime_W_UC_" + str(i) + "_f3")
    h_mT2prime_W_UC_i.Write() 
    c.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f3.pdf") 
    #c2.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f3_LOG.pdf") # logarithmic version of plot 
    #h_mT2prime_W_UC_i.SetAxisRange(50, 120, "X") 
    #c3.SaveAs(outDir + "h_mT2prime_W_UC_" + str(i) + "_f3_LIMITED.pdf") # limited x-axis version of plot 
    h_mT2prime_W_UC_i_f3_data = np.around(np.array([alphaList[i-1], f3.GetParameter(0), f3.GetParameter(1), f3.GetParameter(2), 
                                                    f3.GetParameter(3), f3.GetParameter(4), f3.GetChisquare(), f3.GetNDF(), 
                                                    f3.Derivative(f3.GetParameter(0))]), 3) 
    parameterFile.write('h_mT2prime_W_UC_' + str(i) + '_f3: \t' + str(h_mT2prime_W_UC_i_f3_data) + '\n')
    parameterFile4.write(str(alphaList[i-1]) + ',' + str(f3.GetParameter(0)) + ',' + str(f3.GetParError(0)) + '\n') 

    alpha[0] = alphaList[i-1] 
    par0[0] = f3.GetParameter(0)
    par1[0] = f3.GetParameter(1)
    par2[0] = f3.GetParameter(2) 
    par3[0] = f3.GetParameter(3)  
    par4[0] = f3.GetParameter(4) 
    chisquare[0] = f3.GetChisquare()
    NDF[0] = f3.GetNDF() 
    slope_par0[0] = f3.Derivative(f3.GetParameter(0))
    functionType[0] = 1 
    tree.Fill()

f_outputRoot.Write() 
f_outputRoot.Close()