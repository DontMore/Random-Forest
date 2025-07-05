import pickle
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import numpy as np

# Muat model dari file pickle
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to convert SMILE to numeric
def convert_smile_to_numeric(smile):
    if isinstance(smile, str):
        return len(smile)  # Mengembalikan panjang SMILES sebagai nilai numerik
    else:
        return np.nan  # Handle non-string cases gracefully

# Fungsi untuk menghitung properti molekul
def hitung_properti(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    properti = {
        'Berat Molekul': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'Jumlah Donor Ikatan Hidrogen': Descriptors.NumHDonors(mol),
        'Jumlah Akseptor Ikatan Hidrogen': Descriptors.NumHAcceptors(mol),
        'Jumlah Ikatan yang Dapat Berotasi': Descriptors.NumRotatableBonds(mol),
        'Massa Tepat': Descriptors.ExactMolWt(mol),
        'Massa Monoisotopik': Descriptors.ExactMolWt(mol),
        'Luas Permukaan Polar Topologi': Descriptors.TPSA(mol),
        'Jumlah Atom Berat': Descriptors.HeavyAtomCount(mol),
        'Muatan Formal Total': sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
        'Jumlah Atom Isotop': sum([1 for atom in mol.GetAtoms() if atom.GetIsotope() != 0]),
        'Jumlah Stereosenter Atom Terdefinisi': sum([1 for center in Chem.FindMolChiralCenters(mol, includeUnassigned=True) if center[1] != '?']),
        'Jumlah Stereosenter Atom Tak Terdefinisi': sum([1 for center in Chem.FindMolChiralCenters(mol, includeUnassigned=True) if center[1] == '?']),
        'Jumlah Stereosenter Ikatan Terdefinisi': sum([1 for bond in mol.GetBonds() if bond.GetStereo() in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]]),
        'Jumlah Stereosenter Ikatan Tak Terdefinisi': sum([1 for bond in mol.GetBonds() if bond.GetStereo() not in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREONONE]]),
        'Jumlah Unit yang Terikat Kovalen': len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)),
        'Apakah Senyawa Telah Dikanoikalisasi': (smiles == Chem.MolToSmiles(mol, canonical=True))
    }

    # Hanya menyertakan nilai numerik dalam dataInput
    dataInput = [
        convert_smile_to_numeric(smiles),  #1 Tambahkan panjang SMILES sebagai fitur numerik
        Descriptors.MolWt(mol), #2
        Descriptors.MolLogP(mol), #3
        Descriptors.NumHDonors(mol), #4
        Descriptors.NumHAcceptors(mol), #5
        Descriptors.NumRotatableBonds(mol), #6
        Descriptors.ExactMolWt(mol), #7
        Descriptors.ExactMolWt(mol),
        Descriptors.TPSA(mol), #8
        Descriptors.HeavyAtomCount(mol), #9
        sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),  #10
        sum([1 for atom in mol.GetAtoms() if atom.GetIsotope() != 0]), #11
        sum([1 for center in Chem.FindMolChiralCenters(mol, includeUnassigned=True) if center[1] == '?']), #12
        sum([1 for bond in mol.GetBonds() if bond.GetStereo() in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]]), #13
        sum([1 for bond in mol.GetBonds() if bond.GetStereo() not in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREONONE]]), #14
        len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)), #15
        int(smiles == Chem.MolToSmiles(mol, canonical=True)),  # 17 Konversi boolean ke integer
    ]

    return mol, properti, dataInput

# Fungsi untuk membuat prediksi
def predict(input_data):
    # Pastikan input_data dalam bentuk list of lists
    return model.predict([input_data])

# Antarmuka Streamlit
st.title('Perhitungan Properti Molekul dan Prediksi Bahaya Senyawa Kimia')

# Input notasi SMILES
smiles_input = st.text_input('Masukkan notasi SMILES:', 'CC(=O)O')

if smiles_input:
    # Menghitung properti molekul
    hasil = hitung_properti(smiles_input)
    if hasil is None:
        st.error('Notasi SMILES tidak valid. Silakan masukkan notasi yang benar.')
    else:
        mol, properti, dataInput = hasil  # Menangkap semua nilai yang dikembalikan
        # Menampilkan gambar molekul
        img = Draw.MolToImage(mol)
        st.image(img, caption='Representasi Molekul')
        # Menampilkan properti molekul
        st.subheader('Properti Molekul:')
        for k, v in properti.items():
            st.write(f'**{k}**: {v}')

        # Menyiapkan data input untuk prediksi
        input_data = dataInput  # Gunakan dataInput yang sudah dihitung (hanya numerik)

        # Membuat prediksi
        prediction = predict(input_data)  # Panggil fungsi predict
        st.subheader('Prediksi Bahaya Senyawa Kimia:')
        st.write(f'Prediksi: {prediction}')