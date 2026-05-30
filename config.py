import torch

CANCER_TYPES = [
    "Adrenocortical_carcinoma",
    "Bladder_Urothelial_Carcinoma",
    "Brain_Lower_Grade_Glioma",
    "Breast_invasive_carcinoma",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma",
    "Cholangiocarcinoma",
    "Colon_Rectum_adenocarcinoma",
    "Esophageal_carcinoma",
    "Glioblastoma_multiforme",
    "Head_and_Neck_squamous_cell_carcinoma",
    "Kidney_Chromophobe",
    "Kidney_renal_clear_cell_carcinoma",
    "Kidney_renal_papillary_cell_carcinoma",
    "Liver_hepatocellular_carcinoma",
    "Lung_adenocarcinoma",
    "Lung_squamous_cell_carcinoma",
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
    "Mesothelioma",
    "Ovarian_serous_cystadenocarcinoma",
    "Pancreatic_adenocarcinoma",
    "Pheochromocytoma_and_Paraganglioma",
    "Prostate_adenocarcinoma",
    "Sarcoma",
    "Skin_Cutaneous_Melanoma",
    "Stomach_adenocarcinoma",
    "Testicular_Germ_Cell_Tumors",
    "Thymoma",
    "Thyroid_carcinoma",
    "Uterine_Carcinosarcoma",
    "Uterine_Corpus_Endometrial_Carcinoma",
    "Uveal_Melanoma",
]

NUM_CANCERS = len(CANCER_TYPES)
CANCER_TO_IDX = {name: i for i, name in enumerate(CANCER_TYPES)}

BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 0.01

FREEZE_BACKBONE_EPOCHS = 2

DEVICE = "cpu"
NUM_WORKERS = 4
SEED = 42
