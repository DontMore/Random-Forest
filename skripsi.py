import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

def hitung_properti(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    properti = {
        'Berat Molekul': Descriptors.MolWt(mol),
        'LogP (Crippen)': Crippen.MolLogP(mol),
        'Jumlah Donor Ikatan Hidrogen': Descriptors.NumHDonors(mol),
        'Jumlah Akseptor Ikatan Hidrogen': Descriptors.NumHAcceptors(mol),
        'Jumlah Ikatan yang Dapat Berotasi': Descriptors.NumRotatableBonds(mol),
        'Massa Tepat': Descriptors.ExactMolWt(mol),
        'Luas Permukaan Polar Topologi': Descriptors.TPSA(mol),
        'Jumlah Atom Berat': Descriptors.HeavyAtomCount(mol),
        'Muatan Formal Total': sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
        'Jumlah Atom Isotop': sum([1 for atom in mol.GetAtoms() if atom.GetIsotope() != 0]),
    }

    # Menghitung jumlah stereosenter atom terdefinisi dan tak terdefinisi
    stereo_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    properti['Jumlah Stereosenter Atom Terdefinisi'] = sum([1 for center in stereo_info if center[1] != '?'])
    properti['Jumlah Stereosenter Atom Tak Terdefinisi'] = sum([1 for center in stereo_info if center[1] == '?'])

    # Menghitung jumlah stereosenter ikatan terdefinisi dan tak terdefinisi
    stereo_bond_terdefinisi = 0
    stereo_bond_tak_terdefinisi = 0
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
            if bond.GetStereo() in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]:
                stereo_bond_terdefinisi += 1
            else:
                stereo_bond_tak_terdefinisi += 1

    properti['Jumlah Stereosenter Ikatan Terdefinisi'] = stereo_bond_terdefinisi
    properti['Jumlah Stereosenter Ikatan Tak Terdefinisi'] = stereo_bond_tak_terdefinisi

    # Menghitung jumlah unit yang terikat kovalen
    unit_kovalen = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    properti['Jumlah Unit yang Terikat Kovalen'] = len(unit_kovalen)

    # Menentukan apakah senyawa telah dikanoikalisasi
    smiles_kanonik = Chem.MolToSmiles(mol, canonical=True)
    properti['Apakah Senyawa Telah Dikanoikalisasi'] = (smiles == smiles_kanonik)

    return properti

# Antarmuka Streamlit
st.title('Perhitungan Properti Molekul dari Notasi SMILES')

# Input notasi SMILES
smiles_input = st.text_input('Masukkan notasi SMILES:', 'CC(=O)O')

if smiles_input:
    hasil = hitung_properti(smiles_input)
    if hasil:
        st.subheader('Properti Molekul:')
        for k, v in hasil.items():
            st.write(f'{k}: {v}')
    else:
        st.error('Notasi SMILES tidak valid. Silakan masukkan notasi yang benar.')
