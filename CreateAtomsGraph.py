# CreateAtomsGraph.py
# Author: Cameron Shock
# Date: 7/22


import numpy as np


def init():
    box = np.array([[0.0000,10.0000],[0.0000,10.0000],[0.0000,10.0000]], dtype=float)

    polymers = [[1,2,1,2,1,2,1,2,1,2]]
    bonds = np.array([0,2,1,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,2,1,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,2,1,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,2,1,0],
                     [0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,2],
                     [0,0,0,0,0,0,0,0,0,0])
    angles = np.zeros((10,10,10))
    diehedrals = np.zeros((10,10,10,10))
    numOfPolymers = [2]
    bondTypes = [1,2]
    rigid = [1,0]

    charge = [0,0]
    moment = [0,0.5]
    diameter = [3,3]
    density = [3,3]
    masses = [18,18]

    positions = np.array([np.random.rand((sum(numOfPolymers)))*box[0,1],np.random.rand((sum(numOfPolymers)))*box[1,1],np.random.rand((sum(numOfPolymers)))*box[2,1]]).transpose()
    molecule = np.ones(sum(numOfPolymers))
    required = {'box': box, 'masses': masses, 'polymers': polymers, 'numOfPolymers': numOfPolymers, 'bondTypes': bondTypes, 'rigid': rigid}

    properties = {'position': positions, 'diameter': diameter, 'density':density, 'charge': charge, 'moment': moment, 'molecule': molecule}
    #properties = {'position': positions, 'charge': charge, 'moment': moment, 'molecule': molecule}

    filename = "../testing/atomstest.txt"
    CreateAtoms(required,properties, filename)


def CreateAtoms(required, properties, filename):

    ##### Set Positions
    newpos = np.empty((0,3))
    for i in range(len(required['polymers'])):
        for j in range(required['numOfPolymers'][i]):
            pos = np.array([np.random.rand()*required['box'][0,1], np.random.rand()*required['box'][1,1], np.random.rand()*required['box'][2,1]])
            newpos = np.append(newpos,SetPositions(required['polymers'][i], pos, required['box'],properties['diameter']), axis=0)
    properties['position'] = newpos
    
    ##### Set Bond Types
    dims = CheckDims(required['polymers'])
    bonds = np.empty((0,3))
    if len(required['bondTypes']) != 0:
        if dims == len(required['bondTypes'])+1:
            for i in range(len(required['polymers'])):
                if len(required['polymers'][i]) > 1:
                    bonds = np.append(bonds, CreateBonds(required['polymers'][i],required['numOfPolymers'][i],required['bondTypes']), axis=0)
        else:
            print('Incorrect number of bond types. Should be length ' + str(dims-1) + '.')
            exit()
    # Add bond numbers to first column
    bonds = np.roll(np.c_[bonds, np.array(range(bonds.shape[0]))+1],1,axis=1)


    ##### Set Dipole Moments
    if 'moment' in properties:
        dipoles = np.empty((0,3))
        for i in range(len(required['polymers'])):
            dipoles = np.append(dipoles, SetDipoles(required['polymers'][i],required['numOfPolymers'][i],properties['moment'],required['rigid'], newpos))
        dipoles = dipoles.reshape(int(dipoles.shape[0]/3), 3)
        properties['moment'] = dipoles

    #### Set diameters and densities
    if 'diameter' in properties and 'density' in properties:
        diadens = np.empty((0,2))
        for i in range(len(required['polymers'])):
            for j in range(required['numOfPolymers'][i]):
                diadens = np.append(diadens, SetDiameterDensity(required['polymers'][i], properties['diameter'], properties['density']),axis=0)
        properties['diameter'] = diadens[:,0]
        properties['density'] = diadens[:,1]
    
    #### Set charges
    if 'charge' in properties:
        charge = np.empty((0,1))
        for i in range(len(required['polymers'])):
            for j in range(required['numOfPolymers'][i]):
                charge = np.append(charge, SetCharges(required['polymers'][i], properties['charge']))
        properties['charge'] = charge


    #### Set Molecules
    if 'molecule' in properties:
        molecule = np.empty((0,1))
        mol = 1
        for i in range(len(required['polymers'])):
            for j in range(required['numOfPolymers'][i]):
                    molecule = np.append(molecule, SetMolecule(required['polymers'][i], mol))
                    mol += 1
        properties['molecule'] = molecule

    #### Set Types
    types = np.empty((0,1))
    for i in range(len(required['polymers'])):
        for j in range(required['numOfPolymers'][i]):
            types = np.append(types, SetTypes(required['polymers'][i]))

    ids = np.array(range(types.shape[0]))+1

    PrintFile(filename, required, properties, bonds, types, ids)

    


def SetPositions(polymer, lastpos, box, diameter):
    newpos = []
    for monomer in polymer:
        if type(monomer) == list:
            listpos = SetPositions(monomer, lastpos, box, diameter)
            newpos.extend(listpos)
            lastpos = listpos[0]
        else:
            while True:
                r = diameter[monomer-1]
                phi = 2*np.pi*np.random.rand()
                theta = np.pi*np.random.rand()
                posx = r*np.sin(theta)*np.cos(phi)
                posy = r*np.sin(theta)*np.sin(phi)
                posz = r*np.cos(theta)
                pos = lastpos + np.array([posx,posy,posz])
                if (pos[0] > box[0,0] and pos[1] > box[1,0] and pos[2] > box[2,0] and pos[0] < box[0,1] and pos[1] < box[1,1] and pos[2] < box[2,1]):
                    break
            newpos.append(pos)
            lastpos = pos
        
    return np.array(newpos)


def CheckDims(givenList):
    dims = 1
    for element in givenList:
        newdim = 1
        if type(element) == list:
            newdim += CheckDims(element)
            if newdim > dims:
                dims = newdim
    return dims


def TotalListLength(lists):
    size = 0
    for item in lists:
        if type(item) == list:  
            size += TotalListLength(item)
        else:
         size+=1
    return size


def CreateBonds(polymer, numOfPolymers, bondTypes):
    bonds = []
    dim = CheckDims(polymer)
    length = len(polymer)
    totlength = TotalListLength(polymer)
    for i in range(numOfPolymers):
        mon1 = 1
        mon2 = 2
        j = 1
        for monomer in polymer:
            if type(monomer) == list:
                listBond = CreateBonds(monomer, 1, bondTypes)
                bonds.extend(listBond + np.tile([0, (mon1-1)+(i*totlength), (mon1-1)+(i*totlength)],(listBond.shape[0],1)))
                mon2 = mon1 + listBond.shape[0]+1
                mon1 = listBond[0,1] + mon1-1
            if j < length:
                bonds.append(np.array([bondTypes[dim-1], mon1+(i*totlength), mon2+(i*totlength)]))
            mon1 = mon2
            mon2 += 1
            j += 1
    return np.array(bonds)


def SetDipoles(polymer, numOfPolymer, moments, rigid, pos):
    dipoles = []
    dim = CheckDims(polymer)
    j = 0
    for i in range(numOfPolymer):
        for monomer in polymer:
            if type(monomer) == list:
                dipoles.extend(SetDipoles(monomer, 1, moments, rigid, pos[j:(j+TotalListLength(monomer))]))
                j += 1
            else:
                if rigid[dim-1] == 1 and len(polymer) > 1:
                    r = moments[polymer[1]-1]
                    norm = np.linalg.norm(pos[1]-pos[0])
                    print(pos[0])
                    mx = r*(pos[1,0] - pos[0,0])/norm
                    my = r*(pos[1,1] - pos[0,1])/norm
                    mz = r*(pos[1,2] - pos[0,2])/norm
                    dipoles.append(np.array([0,0,0]))
                    dipoles.append(np.array([mx,my,mz]))
                    break
                else:
                    r = moments[monomer-1]
                    phi = 2*np.pi*np.random.rand()
                    theta = np.pi*np.random.rand()
                    mx = r*np.sin(theta)*np.cos(phi)
                    my = r*np.sin(theta)*np.sin(phi)
                    mz = r*np.cos(theta)
                    dipoles.append(np.array([mx,my,mz]))
            j += 1
    return np.array(dipoles)


def SetDiameterDensity(polymer, diameters, densities):
    diadens = []
    for monomer in polymer:
        if type(monomer) == list:
            diadens.extend(SetDiameterDensity(monomer, diameters, densities))
        else:
            diadens.append([diameters[monomer-1], densities[monomer-1]])
    return np.array(diadens)


def SetCharges(polymer, charge):
    charges = []
    for monomer in polymer:
        if type(monomer) == list:
            charges.extend(SetCharges(monomer, charge))
        else:
            charges.append([charge[monomer-1]])
    return np.array(charges)


def SetMolecule(polymer, val):
    molecules = []
    for monomer in polymer:
        if type(monomer) == list:
            molecules.extend(SetMolecule(monomer, val))
            val += 1
        else:
            molecules.append([val])
    return np.array(molecules)

def SetTypes(polymer):
    types = []
    for monomer in polymer:
        if type(monomer) == list:
            types.extend(SetTypes(monomer))
        else:
            types.append(monomer)
    return np.array(types)



def PrintFile(filename, required, properties, bonds, types, ids):
    file = open(filename, 'w+')
    file.write("#" + filename + "\n")
    file.write(str(int(ids.shape[0])) + "\tatoms\n")
    file.write(str(int(bonds[-1,0])) + "\tbonds\n\n")

    file.write(str(len(required['masses'])) + "\tatom types\n")
    file.write(str(len(required['bondTypes'])) + "\tbond types\n\n")

    file.write('{0:.5f}'.format(required['box'][0,0]) + "\t" + '{0:.5f}'.format(required['box'][0,1]) + "\txlo xhi\n")
    file.write('{0:.5f}'.format(required['box'][1,0]) + "\t" + '{0:.5f}'.format(required['box'][1,1]) + "\tylo yhi\n")
    file.write('{0:.5f}'.format(required['box'][2,0]) + "\t" + '{0:.5f}'.format(required['box'][2,1]) + "\tzlo zhi\n\n\n\n")

    file.write("Masses\n\n")
    i = 1
    for mass in required['masses']:
        file.write(str(i) + "\t" + str(mass) + "\n")
    
    fields = list(properties.keys())
    file.write("\n\n\nAtoms")
    file.write("\t# id type ")
    for field in fields:
        if field == 'position':
            file.write("x y z ")
        elif field == 'moment':
            file.write("mx my mz ")
        else:
            file.write(field + " ")
    file.write("\n\n")
        

    for i in range(ids.shape[0]):
        file.write(str(int(ids[i])) + "\t" + str(int(types[i])))
        for field in fields:
            if type(properties[field][i]) == np.ndarray:
                for item in properties[field][i]:
                    file.write("\t" + '{0:.5f}'.format(item))
            else:
                if field == 'molecule':
                    file.write("\t" + str(int(properties[field][i])))
                else:
                    file.write("\t" + '{0:.5f}'.format(properties[field][i],5))
        file.write("\n")

    if bonds.shape[0] > 0:
        file.write("\n\n\nBonds\n\n")

        for i in range(bonds.shape[0]):
            for item in bonds[i]:
                file.write(str(int(item)) + "\t")
            file.write("\n")

    file.close()


init()