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

#include "PrimaryGeneratorAction.hh"

#include "PrimaryGeneratorMessenger.hh"
#include "DetectorConstruction.hh"

#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "Randomize.hh"

#define G4ExpoRand() CLHEP::RandExponential::shoot(2.7)

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::PrimaryGeneratorAction(DetectorConstruction* det)
: Detector(det)
{
  G4int n_particle = 1;
  particleGun  = new G4ParticleGun(n_particle);
  SetDefaultKinematic();
  beam = 0.*mm;
  
  //create a messenger for this class
  gunMessenger = new PrimaryGeneratorMessenger(this);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete particleGun;
  delete gunMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4double PrimaryGeneratorAction::GetX0() const
{
  return x0 + (Detector->GetCalorSizeXY() - 1.) / 2.; // + Detector->GetCalorThickness() * sin(theta) * cos(phi);
}

G4double PrimaryGeneratorAction::GetY0() const
{
  return y0 + (Detector->GetCalorSizeXY() - 1.) / 2.; // + Detector->GetCalorThickness() * sin(theta) * sin(phi);
}

G4double PrimaryGeneratorAction::GetSizeXY() const
{
  return Detector->GetCalorSizeXY();
}

G4double PrimaryGeneratorAction::GetSizeZ() const
{
  return Detector->GetCalorThickness();
}

void PrimaryGeneratorAction::SetDefaultKinematic()
{
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition* particle = particleTable->FindParticle(particleName="mu-");
  particleGun->SetParticleDefinition(particle);
  theta = 2. * M_PI / 2. * (G4UniformRand() - 0.5); // angle to vertical
  phi = 2. * M_PI / 2. * (G4UniformRand() - 0.5);
  G4cout << "theta " << theta << ", phi " << phi << G4endl;
  x0 = 2 * 24 * (G4UniformRand() - 0.5); // x at the top of the ecal
  y0 = 2 * 24 * (G4UniformRand() - 0.5); // y at the top of the ecal
  particleGun->SetParticleMomentumDirection(G4ThreeVector(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)));
  particleGun->SetParticleEnergy(100.*GeV);
  G4double position = 0.5*(Detector->GetWorldSizeZ());
  particleGun->SetParticlePosition(G4ThreeVector(x0 * mm, y0 * mm, position));

  // G4ThreeVector position = particleGun->GetParticlePosition();
  // x0 = 0.; // x at the top of the ecal
  // y0 = 0.; // y at the top of the ecal
  // G4double z0 = position.z();
  // theta = 0.; // angle to vertical
  // phi = 0.;
  // particleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
  // particleGun->SetParticleMomentumDirection(G4ThreeVector(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)));
  // particleGun->SetParticleEnergy(106 * MeV);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PrimaryGeneratorAction::SetRandomCosmic()
{
  G4double maxXY = 0.49 * (Detector->GetCalorSizeXY());
  x0 = 2. * maxXY * (G4UniformRand() - 0.5) / 2.; // x at the top of the ecal
  y0 = 2. * maxXY * (G4UniformRand() - 0.5) / 2.; // y at the top of the ecal
  z0 = particleGun->GetParticlePosition().z();
  if (std::abs(x0) > maxXY)
  {
    x0 = maxXY;
  }
  if (std::abs(y0) > maxXY)
  {
    y0 = maxXY;
  }
  theta = 2. * atan(maxXY / z0) * (G4UniformRand() - 0.5); // angle to vertical
  phi = 2. * atan(maxXY / z0) * (G4UniformRand() - 0.5);
  // G4double max_energy = 100; // in GeV
  // G4double min_energy = 1;   // in GeV
  // energy = 1.0 * GeV * exp(log(min_energy) + G4UniformRand() * (log(max_energy / min_energy))); // old, for cosmics
  energy = 4 * GeV + 0.14 * GeV * G4ExpoRand(); // for cosmics
}

void PrimaryGeneratorAction::SetDecay()
{
  x0 = 5.5 * 2 * (G4UniformRand() - 0.5) * cm;;
  y0 = 5.5 * 2 * (G4UniformRand() - 0.5) * cm;;
  z0 = 5.5 * 2 * (G4UniformRand() - 0.5) * cm; // 2 * parenthesis is uniform distribution between -1 and +1
  theta = 2 * M_PI * 2 * (G4UniformRand() - 0.5);
  phi = 2 * M_PI * 2 * (G4UniformRand() - 0.5);
  energy = 107. * MeV; // for decays
}

void PrimaryGeneratorAction::SetMIP()
{
  G4ThreeVector position = particleGun->GetParticlePosition();
  G4double maxXY = 0.49 * (Detector->GetCalorSizeXY());
  x0 = 2. * maxXY * (G4UniformRand() - 0.5) / 2.; // x at the top of the ecal
  y0 = 2. * maxXY * (G4UniformRand() - 0.5) / 2.; // y at the top of the ecal
  z0 = position.z();
  if (std::abs(x0) > maxXY)
  {
    x0 = maxXY;
  }
  if (std::abs(y0) > maxXY)
  {
    y0 = maxXY;
  }
  G4double max_theta = atan(maxXY/(z0-Detector->GetCalorThickness()/2.));
  theta = max_theta * 2 * (G4UniformRand() - 0.5);
  phi = 2 * M_PI * 2 * (G4UniformRand() - 0.5);
  energy = 4. * GeV; // for MIPs
}

void PrimaryGeneratorAction::SetPhaseSpaceScan()
{
  // G4ThreeVector position = particleGun->GetParticlePosition();
  // z0 = position.z();
  // G4double theta_x = 1.4 * 2 * (G4UniformRand() - 0.5); // -80° to 80°
  // G4double theta_y = 1.4 * 2 * (G4UniformRand() - 0.5); // -80° to 80°
  // theta = atan(sqrt(tan(theta_x) * tan(theta_x) + tan(theta_y) * tan(theta_y)));
  // phi = acos(tan(theta_x)/tan(theta));
  // x0 = z0 * tan(theta_x);
  // y0 = z0 * tan(theta_y);
  // energy = 4. * GeV;
  G4double maxXY = Detector->GetCalorSizeXY();
  x0 = 2. * maxXY * (G4UniformRand() - 0.5) * 2.; // x at the top of the ecal
  y0 = 2. * maxXY * (G4UniformRand() - 0.5) * 2.; // y at the top of the ecal
  z0 = 50. * mm + 0.5 * Detector->GetCalorThickness();
  if (x0 > 0) {
    theta = -atan(sqrt(x0*x0 + y0*y0)/z0);
  } else {
    theta = atan(sqrt(x0 * x0 + y0 * y0) / z0);
  }
  phi = atan(y0/x0);
  energy = 4. * GeV; 
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event *anEvent)
{
  //this function is called at the begining of event
  //
  //randomize beam, if requested.
  //
  SetPhaseSpaceScan();
  particleGun->SetParticlePosition(G4ThreeVector(x0, y0, z0));
  particleGun->SetParticleMomentumDirection(G4ThreeVector(sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)));
  particleGun->SetParticleEnergy(energy);
  particleGun->GeneratePrimaryVertex(anEvent);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

