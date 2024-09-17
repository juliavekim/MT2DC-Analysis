###############################################
# Mt2dc fitting analysis:
#The second plotting code, which produces graphs used for the study of which $\alpha$ gives the best mass measurement. These graphs plot (1) the first fitted parameter and (2) its error for both fits 1 and 2 as a function of $\alpha.$ 
##############################################

import ROOT
import numpy as np 

##############################################
# Define the input and output root files
##############################################
alpha = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f1_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 0)
mass_fit1 = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f1_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 1)
mass_fit1_ucty = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f1_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 2)
mass_fit2 = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f2_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 1)
mass_fit2_ucty = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f2_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 2)
mass_fit3 = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f3_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 1)
mass_fit3_ucty = np.loadtxt("/Users/juliakim/Documents/styledPlotsOutputs/mT2prime_W_UC_f3_Parameter0.txt", unpack=True, skiprows = 1, delimiter=',', usecols = 2)

outDir = "/Users/juliakim/Documents/styledPlotsOutputs/"
f_outputRoot = ROOT.TFile.Open("/Users/juliakim/Documents/mt2dc_alpha_plotter_output.root", "recreate")

##############################################
# Produce graphs 
##############################################
c = ROOT.TCanvas()
gr = ROOT.TGraph(len(alpha), alpha, mass_fit1) 
gr.Draw("AC*") 
gr.SetTitle("mass_fit1(W) vs. #alpha; #alpha; mass_fit1(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f1_Par0.pdf")
gr.Write() 

gr2 = ROOT.TGraph(len(alpha), alpha, mass_fit1_ucty) 
gr2.Draw("AC*") 
gr2.SetTitle("mass_fit1_ucty(W) vs. #alpha; #alpha; mass_fit1_ucty(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f1_Par0err.pdf") 
gr2.Write()

gr3 = ROOT.TGraph(len(alpha), alpha, mass_fit2) 
gr3.Draw("AC*") 
gr3.SetTitle("mass_fit2(W) vs. #alpha; #alpha; mass_fit2(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f2_Par0.pdf") 
gr3.Write() 

gr4 = ROOT.TGraph(len(alpha), alpha, mass_fit2_ucty) 
gr4.Draw("AC*") 
gr4.SetTitle("mass_fit2_ucty(W) vs. #alpha; #alpha; mass_fit2(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f2_Par0err.pdf") 
gr4.Write() 

gr5 = ROOT.TGraph(len(alpha), alpha, mass_fit3) 
gr5.Draw("AC*") 
gr5.SetTitle("mass_fit3(W) vs. #alpha; #alpha; mass_fit3(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f3_Par0.pdf") 
gr5.Write() 

gr6 = ROOT.TGraph(len(alpha), alpha, mass_fit3_ucty) 
gr6.Draw("AC*") 
gr6.SetTitle("mass_fit3_ucty(W) vs. #alpha; #alpha; mass_fit2(W) [GeV]") 
c.SaveAs(outDir + "g_mT2prime_W_UC_f3_Par0err.pdf") 
gr6.Write() 

f_outputRoot.Close()
