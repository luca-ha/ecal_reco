//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file electromagnetic/TestEm1/src/HistoManager.cc
/// \brief Implementation of the HistoManager class
//
// 
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "HistoManager.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

HistoManager::HistoManager()
  : fFileName("ecal")
{
  Book();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

HistoManager::~HistoManager()
{
  // auto analysisManager = G4AnalysisManager::Instance();
  // analysisManager->Write();
  // analysisManager->CloseFile(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HistoManager::Book()
{
  // // Create or get analysis manager
  // // The choice of analysis technology is done via selection of a namespace
  // // in HistoManager.hh
  // G4AnalysisManager* analysisManager = G4AnalysisManager::Instance();
  // analysisManager->SetDefaultFileType("root");
  // analysisManager->SetFileName(fFileName);
  // analysisManager->SetVerboseLevel(1);

  // // create ROOT tree
  // analysisManager->CreateNtuple("events", "recorded info per event");
  // analysisManager->CreateNtupleDColumn(0, "E"); // 0
  // analysisManager->CreateNtupleIColumn(0, "pdg"); // 1
  // analysisManager->CreateNtupleDColumn(0, "HCEnergyVector", fEventAction->GetEcalEdep()); // 2
  // analysisManager->CreateNtupleDColumn(0, "detid"); // 3
  // analysisManager->FinishNtuple(0);


  // analysisManager->SetActivation(true);    // enable inactivation of histograms

  // // Define histograms start values
  // const G4int kMaxHisto = 6;
  // const G4String id[] = {"dummy", "Ecal_tot", "Evis_tot", "Etot_profile" , "Evis_rofile", "Evis_scint"};
  // const G4String title[] = 
  //               { "dummy",                    //0 
	// 			  "total Etot in Ecal",       //1
  //                 "total Evis in Ecal",       //2
  //                 "Etot profile",             //3
  //                 "Evis profile",             //4
  //                 "Evis per scint"            //5					  
  //                };
				 
  // // Default values (to be reset via /analysis/h1/set command)               
  // G4int nbins = 100;
  // G4double vmin = 0.;
  // G4double vmax = 100.;

  // // Create all histograms as inactivated 
  // // as we have not yet set nbins, vmin, vmax
  // for (G4int k=0; k<kMaxHisto; k++) {
  //   G4int ih = analysisManager->CreateH1(id[k], title[k], nbins, vmin, vmax);
  //   analysisManager->SetH1Activation(ih, false);
  // }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
