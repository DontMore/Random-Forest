import streamlit as st
import numpy as np
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PIL import Image
import io

# === Memuat Model === #
@st.cache_data
def load_model():
    with open("best_rf_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return model

rf_model = load_model()

# --- Fungsi untuk ekstrak fitur dari SMILES (harus sama seperti saat training) ---
def extract_features_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Hitung ikatan
    num_single = num_double = num_triple = num_aromatic = num_ring = 0
    for bond in mol.GetBonds():
        btype = bond.GetBondType()
        if btype.name == "SINGLE": num_single += 1
        elif btype.name == "DOUBLE": num_double += 1
        elif btype.name == "TRIPLE": num_triple += 1
        elif btype.name == "AROMATIC": num_aromatic += 1
        if bond.IsInRing(): num_ring += 1

    # Fitur yang digunakan saat training (7 fitur penting)
    features = pd.DataFrame([{
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumSingleBonds": num_single,
        "NumBonds": mol.GetNumBonds(),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol)
    }])
    
    feature_dict = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumSingleBonds": num_single,
        "NumBonds": mol.GetNumBonds(),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumDoubleBonds": num_double,
        "NumTripleBonds": num_triple,
        "NumAromaticBonds": num_aromatic,
        "NumRingBonds": num_ring
    }
    
    return features, feature_dict

# === UI Streamlit === #
st.title("🧪 Prediksi Bahaya Senyawa dari Notasi SMILES")
st.write("Masukkan notasi **SMILES** untuk menghitung fitur molekuler dan melakukan klasifikasi multi-label.")

# Input SMILES dari pengguna
smiles_input = st.text_input("Masukkan SMILES", "CN(C)C(=N)N=C(N)N")

if st.button("Prediksi"):
    features, feature_dict = extract_features_from_smiles(smiles_input)
    
    if features is None:
        st.error("⚠️ Notasi SMILES tidak valid. Coba lagi!")
    else:
        # Tampilkan struktur molekul 2D
        st.subheader("🔮 Visualisasi Struktur Molekul 2D")
        mol = Chem.MolFromSmiles(smiles_input)
        img = Draw.MolToImage(mol)
        st.image(img)
        
        # Tampilkan fitur molekuler dalam tabel
        st.subheader("🔬 Hasil Perhitungan Fitur Molekuler")
        feature_df = pd.DataFrame([feature_dict])  # Membuat dataframe dengan satu baris
        st.dataframe(feature_df.style.set_properties(**{'text-align': 'center'}))
        
        # Prediksi dengan model
        # Note: You might need to adjust this line to match your model's expected input format
        prediction = rf_model.predict(features)
        
        # Label multi-class dan keterangannya
        labels = [
            ("Corrosive", "Dapat menyebabkan korosi atau iritasi"),
            ("Irritant", "Menimbulkan iritasi pada kulit atau mata"),
            ("Acute Toxic", "Beracun dalam paparan singkat"),
            ("Health Hazard", "Berbahaya bagi kesehatan dalam jangka panjang"),
            ("Environmental Hazard", "Berbahaya bagi lingkungan"),
            ("Flammable", "Mudah terbakar"),
            ("Compressed Gas", "Gas bertekanan tinggi yang dapat meledak"),
            ("Oxidizer", "Bahan pengoksidasi yang dapat menyebabkan kebakaran"),
        ]

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        predicted_data = [(label, desc) for i, (label, desc) in enumerate(labels) if prediction[0][i] == 1]

        if predicted_data:
            pred_df = pd.DataFrame(predicted_data, columns=["Label", "Keterangan"])
            st.table(pred_df)
        else:
            st.warning("Tidak ada label yang terdeteksi untuk sampel ini.")