import sys
sys.path.insert(1, 'graphnn/')

from scripts.get_qm9 import download, Molecule, string_convert

import numpy as np
import ase
import ase.db
import ase.data
import zipfile

# Converts the zip file containing the PC9 data into a single XYZ file.
def zip_to_xyz(zippath, dest):
    zip = zipfile.ZipFile(zippath, mode="r")
    inf_list = zip.infolist()
    inf_list = [inf for inf in inf_list if inf.filename[0:len("PC9_data/XYZ/")] == "PC9_data/XYZ/" and len(inf.filename) > len("PC9_data/XYZ/")]
    with open(dest, "wb") as f:
        for zipinfo in inf_list:
            zipf = zip.open(zipinfo)
            f.write(zipf.read())

######################################################################################
# Code modified from the graphh package by Peter Bjørn Jørgensen (pbjo@dtu.dk) et al.
######################################################################################
# Modified to fit the PC9 formatting
def load_xyz_file(filename):
    predefined_keys = """tag
    index
    homom1
    lumop1
    A_
    B_
    C_
    homo
    lumo
    gap
    D_
    E_
    E
    F_
    G_
    H_
    I_""".split()
    STATE_READ_NUMBER = 0
    STATE_READ_COMMENT = 1
    STATE_READ_ENTRY = 2
    STATE_READ_NUM_HEAVY = 3 # Replaces the frequency line in the QM9 dataset.
    STATE_READ_SMILES = 4
    STATE_READ_INCHI = 5
    STATE_FAILURE = 6

    state = STATE_READ_NUMBER
    entries_read = 0
    cur_desc = None

    with open(filename, "r") as f:
        for line_no, line in enumerate(f):
            try:
                if state == STATE_READ_NUMBER:
                    entries_to_read = int(line)
                    cur_desc = Molecule(entries_to_read)
                    entries_read = 0
                    state = STATE_READ_COMMENT
                elif state == STATE_READ_COMMENT:
                    # Read comment as whitespace separated values
                    for key, value in zip(predefined_keys, line.split()):
                        if hasattr(cur_desc, key):
                            raise KeyError(
                                "Molecule already contains property %s" % key
                            )
                        else:
                            setattr(cur_desc, key.strip(), string_convert(value))
                    state = STATE_READ_ENTRY
                elif state == STATE_READ_ENTRY:
                    parts = line.split()
                    assert len(parts) == 5
                    atom = parts[0]
                    el_number = ase.data.atomic_numbers[atom]
                    strat_parts = map(lambda x: x.replace("*^", "E"), parts[1:4])
                    floats = list(map(float, strat_parts))
                    cur_desc.coord[entries_read, :] = np.array(floats)
                    cur_desc.z[entries_read] = el_number
                    entries_read += 1
                    if entries_read == cur_desc.z.size:
                        state = STATE_READ_NUM_HEAVY
                elif state == STATE_READ_NUM_HEAVY:
                    cur_desc.num_heavy = np.array(
                        list(map(string_convert,line.split()))
                    )
                    state = STATE_READ_SMILES
                elif state == STATE_READ_SMILES:
                    cur_desc.smiles = line.split()
                    state = STATE_READ_INCHI
                elif state == STATE_READ_INCHI:
                    cur_desc.inchi = line.split()
                    yield cur_desc
                    state = STATE_READ_NUMBER
                elif state == STATE_FAILURE:
                    entries_to_read = None
                    try:
                        entries_to_read = int(line)
                    except:
                        pass
                    if entries_to_read is not None:
                        print("Resuming parsing on line %d" % line_no)
                        cur_desc = Molecule(entries_to_read)
                        entries_read = 0
                        state = STATE_READ_COMMENT
                else:
                    raise Exception("Invalid state")
            except Exception as e:
                print("Exception occured on line %d: %s" % (line_no, str(e)))
                state = STATE_FAILURE

######################################################################################
# Code modified from the graphh package by Peter Bjørn Jørgensen (pbjo@dtu.dk) et al.
######################################################################################
# Modified to fit the PC9 formatting.
def xyz_to_ase(filename, output_name):
    """
    Convert xyz descriptors to ase database
    """

    """
    =================================
      Ele-    ZPVE         U (0 K)   
      ment   Hartree       Hartree   
    =================================
       H     0.000000     -0.500273  
       C     0.000000    -37.846772  
       N     0.000000    -54.583861  
       O     0.000000    -75.064579  
       F     0.000000    -99.718730  
    =================================
    """
    HARTREE_TO_EV = 27.21138602
    REFERENCE_DICT = {
        ase.data.atomic_numbers["H"]: {
            "U0": -0.500273,
        },
        ase.data.atomic_numbers["C"]: {
            "U0": -37.846772,
        },
        ase.data.atomic_numbers["N"]: {
            "U0": -54.583861,
        },
        ase.data.atomic_numbers["O"]: {
            "U0": -75.064579,
        },
        ase.data.atomic_numbers["F"]: {
            "U0": -99.718730,
        },
    }

    # Make a transposed dictionary such that first dimension is property
    REFERENCE_DICT_T = {}
    atom_nums = [ase.data.atomic_numbers[x] for x in ["H", "C", "N", "O", "F"]]
    prop_dict = dict(zip(atom_nums, [REFERENCE_DICT[at]["U0"] for at in atom_nums]))
    REFERENCE_DICT_T["E"] = prop_dict # Treats the U0 as E

    # List of tag, whether to convert hartree to eV
    keywords = [
        ["tag", False],
        ["index", False],
        ["homom1", True], # homo-1
        ["lumop1", True], # lumo+1
        ["A_", False],
        ["B_", False],
        ["C_", False],
        ["homo", True],
        ["lumo", True],
        ["gap", True],
        ["D_", False],
        ["E_", False],
        ["E", True],
        ["F_", False],
        ["G_", False],
        ["H_", False],
        ["I_", False],
    ]
    # Load xyz file
    descriptors = load_xyz_file(filename)

    with ase.db.connect(output_name, append=False) as asedb:
        properties_dict = {}
        for desc in descriptors:
            # Convert attributes to dictionary and convert hartree to eV
            for key, convert in keywords:
                properties_dict[key] = getattr(desc, key)
                # Subtract reference energies for each atom
                if key in REFERENCE_DICT_T:
                    for atom_num in desc.z:
                        properties_dict[key] -= REFERENCE_DICT_T[key][atom_num]
                if convert:
                    properties_dict[key] *= HARTREE_TO_EV
            atoms = ase.Atoms(numbers=desc.z, positions=desc.coord, pbc=False)
            asedb.write(atoms, key_value_pairs=properties_dict)

if __name__ == "__main__":
    url = "https://figshare.com/ndownloader/files/16526123"
    filename = "data/pc9.zip"
    xyz_name = "data/pc9.xyz"
    final_dest = "data/pc9.db"
    print("downloading dataset...")
    download(url, filename)
    print("extracting...")
    zip_to_xyz(filename, xyz_name)
    print("writing to ASE database...")
    xyz_to_ase(xyz_name, final_dest)
    print("done")
