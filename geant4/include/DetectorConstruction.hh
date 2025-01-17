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
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"
#include "G4Cache.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;
class G4Material;
class G4GlobalMagFieldMessenger;
            
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
  
  DetectorConstruction();
 ~DetectorConstruction();

public:
     
  virtual G4VPhysicalVolume* Construct();
  virtual void ConstructSDandField();
  
  void PrintCalorParameters();
       
public:

  G4VPhysicalVolume* GetPvolWorld()         {return pvol_world;};
  G4Material*        GetWorldMaterial()     {return worldMat;};
  G4double           GetWorldSizeZ()        {return worldSizeZ;};
  G4double           GetCalorThickness()    {return calorThickness;};  
  G4double           GetCalorSizeXY()       {return scintLength;};
  G4double           GetModuleThickness()   {return moduleThickness;};
	
  G4LogicalVolume*   GetLvolScint()         {return lvol_scint;};
  G4LogicalVolume*   GetLvolLayer()         {return lvol_layer;};  	
  G4LogicalVolume*   GetLvolModule()        {return lvol_module;};
  G4LogicalVolume*   GetLvolCalorimeter()   {return lvol_calorimeter;};
  G4LogicalVolume*   GetLvolWorld()         {return lvol_world;};  
  
  G4int              GetNbScints()          {return nbOfScints;};  
  G4int              GetNbLayers()          {return nbOfLayers;};    
  G4int              GetNbModules()         {return nbOfModules;};
        			 
private:

  //scintillator
  //
  G4Material*      scintMat;
  G4double scintDiameter, scintWidth, scintLength, scintHeight;
  G4LogicalVolume* lvol_scint;
  
  //layers
  //
  G4Material*      absorberMat;
  G4int            nbOfScints;
  G4double         layerThickness;
  G4LogicalVolume* lvol_layer;
  G4double leadThickness;
  G4LogicalVolume* lvol_lead;
  G4LogicalVolume *lvol_Al_layer;
  G4double        aluThickness;
  G4double         gapSize;

      // modules
      //
      G4Material *moduleMat;
  G4int            nbOfLayers;
  G4double         moduleThickness;    
  G4LogicalVolume* lvol_module;  
           
  //calorimeter
  //
  G4Material*      calorimeterMat;  
  G4int            nbOfModules;
  G4double         calorThickness;
  G4LogicalVolume* lvol_calorimeter;
  G4Material*      shieldingMat;
  G4double         shieldThickness;
  G4LogicalVolume* lvol_topPlate;
  G4LogicalVolume* lvol_sidePlateXZ;
  G4LogicalVolume* lvol_sidePlateYZ;

  //world
  //
  G4Material*        worldMat;
  G4double           worldSizeZ;
  G4LogicalVolume*   lvol_world;                
  G4VPhysicalVolume* pvol_world;
  
  G4Material*        defaultMat;
              
  G4Cache<G4GlobalMagFieldMessenger*> fFieldMessenger;  
      
private:

  void DefineMaterials();
  G4VPhysicalVolume* ConstructCalorimeter();
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

