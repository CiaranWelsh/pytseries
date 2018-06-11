import unittest
from go import *




features = {'RHOB', 'SERPINE1.1', 'RHOB.1', 'DNAJB5', 'S1PR5',
            'IL6', 'NR2F2.2', 'TMEM2', 'TSPAN2', 'ARRDC4.1', 'RGS4',
            'RGS3', 'ADM', 'KLF10', 'PDE4B', 'SMIM13', 'TRIB1',
            'PCDH18.2', 'PFKFB4', 'MURC.1', 'GLI2.2', 'PGM2L1',
            'GADD45B', 'NKX3-1', 'BDNF', 'SNAI1', 'SOX4.3', 'SEMA7A',
            'CCNG2', 'TBX3', 'TXNIP.3', 'ESM1.1', 'NGF', 'RNF152',
            'FHL3', 'PHLDA1', 'FAM43A', 'EGR2.2', 'FHOD1', 'CLDN4',
            'DLX2', 'MYO10', 'F3', 'TXNIP.1', 'DUSP6.3', 'NR2F2.1',
            'HBEGF', 'ARRDC4.2', 'RBMS1', 'CD3EAP', 'CXCL8', 'OSR2.2',
            'ADAMTS1.1', 'DACT1', 'LIF.2', 'MALAT1', 'ZSWIM6.1',
            'RARA.1', 'PLAUR', 'FSTL3', 'SKIL.2', 'COMP', 'GAL.1',
            'PMEPA1', 'IRF2BP2.1', 'ARHGEF40', 'ARRDC4', 'KCTD11.1',
            'ID3', 'COL7A1', 'TXNIP', 'ENPP1.1', 'CTPS1', 'USP53.1',
            'ZNF365', 'HOXD10', 'TIPARP.1', 'KCTD11', 'JUN.2', 'KANK4',
            'PCDH18.1', 'CSRNP1.1', 'ARRDC3.1', 'NABP1', 'NEDD9.2',
            'ARHGAP29', 'GLI2.1', 'PLK3', 'TRIM32', 'JUN.3', 'CCL2',
            'CDC42EP3.1', 'COL16A1', 'SERPINE1', 'ARID5B', 'UGCG',
            'SH3PXD2A.1', 'DLC1', 'VDR', 'AUTS2', 'ARRDC3', 'BLOC1S2',
            'NR4A1', 'GCNT1.3', 'SOWAHC', 'ST6GAL2', 'UCN2', 'DUSP6',
            'TRIB2', 'TSPAN13', 'DUSP6.1', 'NR2F2', 'EGR1.1', 'SOX4.2',
            'GDF6', 'EDN1.1', 'HIC1', 'STK38L.1', 'TRIB2.1', 'HMOX1.2',
            'BCOR', 'VEGFA', 'MEX3B.1', 'KLF10.1', 'SMURF1', 'LMCD1',
            'AMIGO2.2', 'RGS3.1', 'SLC19A2.2', 'GCNT1.1', 'PTX3',
            'PTGS2.2', 'SH3PXD2A.2', 'EGR3', 'FAM46B', 'TFAP2A.1',
            'EGR2.1', 'CLCF1.1', 'NUAK1', 'NR4A1.2', 'ULK1', 'EGR1.2',
            'TIPARP', 'CLCF1.2', 'DDIT4', 'PDGFA.2', 'SLC19A2.1',
            'DUSP1', 'TSHZ3.1', 'HES1.1', 'SMOX.3', 'USP53', 'SOX9',
            'BCOR.1', 'CDKN2B', 'C5orf30', 'CDC42SE1', 'NEDD9',
            'PNRC1', 'NET1.2', 'TSHZ3', 'LIF', 'KIT', 'FOXO1',
            'FAM110B', 'SGK223', 'TLR4', 'KLF2', 'SERPINE1.2',
            'IER2', 'C3orf52', 'C3orf52.1', 'CXCL1', 'OSR2.3',
            'NUAK1.1', 'VEGFA.1', 'ID4.1', 'STK38L.3', 'DUSP5',
            'NET1.1', 'STK38L', 'VASN', 'PFKFB3.3', 'SOX4.5',
            'BHLHE40', 'IRF2BPL', 'ENC1', 'IL11', 'HEY1', 'SLC19A2',
            'SMAD7', 'SUSD6', 'TMEM41B.2', 'HAS2.2', 'RHOB.3',
            'LRRC8C', 'HAS2.1', 'EGR1.3', 'SKIL.1', 'LHX9', 'SKI.1',
            'CXXC5', 'EDN1', 'FOSB', 'ESM1', 'TSPAN2.2',
            'TGFBI.2', 'NUAK1.2', 'PTGS2', 'GLI2', 'TWIST1',
            'ADM.1', 'PCDH18.3', 'PALM2', 'SOWAHC.1',
            'TFAP2A', 'BCL11A', 'JUN.1', 'IFFO2', 'LINC00312',
            'PXDC1', 'ID1', 'VEGFA.7', 'PPP1R3B', 'KBTBD6', 'ID1.1',
            'TNFAIP6', 'SH3PXD2A', 'NIPAL4', 'TMEM2.1', 'GADD45B.2',
            'MYC', 'DUSP1.2', 'IRF2BP2.2', 'RYBP', 'EGR2', 'KLF2.1',
            'ZSWIM6', 'TLR4.1', 'RGS4.2', 'DUSP2', 'OSR2', 'SOCS2', 'ID2',
            'DLC1.1', 'CITED2', 'IL11.1', 'CSRNP1', 'ID4', 'PMEPA1.2',
            'CITED2.1', 'CRISPLD2', 'SPHK1', 'PMEPA1.1', 'JUN', 'CTGF.1',
            'IRF2BP2', 'PRICKLE1', 'FOS.1', 'MEDAG', 'YRDC', 'CDC42EP3',
            'BDNF.3', 'LARP6', 'STC2', 'TUFT1', 'PLEK2', 'TGFBI.1',
            'PFKFB3.1', 'RGS4.3', 'BEND7', 'IFIT2', 'KLF7.2', 'IRF2BPL.1',
            'LIF.1', 'PUS1', 'PFKFB3.2', 'PTGS2.1', 'VDR.1', 'RGS2', 'FN1',
            'ITPRIP', 'SERTAD1', 'PGM2L1.2', 'EGR1', 'HES1', 'OSR2.1',
            'CEBPD.2', 'CEBPD.1', 'MURC', 'IER3', 'SMOX', 'NET1', 'ERRFI1',
            'LMCD1.3', 'PDGFA', 'TGIF1', 'IER5L', 'SKIL', 'SMOX.1', 'BLOC1S2.1',
            'VEGFA.4', 'DUSP6.2', 'TMEM41B', 'NPR3', 'CIRBP', 'PFKFB3',
            'VEGFA.3', 'TMEM41B.1', 'GCNT1.2', 'SIX1', 'PER2', 'SOX4', 'KLF7',
            'PGM2L1.1', 'CLDN4.1', 'ZNF469', 'HMOX1', 'KLF7.1', 'TUFT1.1', 'SOX4.4',
            'ZNF362', 'NEDD9.1', 'HBEGF.1', 'INHBA', 'BDNF.2', 'ELMSAN1', 'PDGFA.1',
            'ADO', 'AMIGO2', 'BDNF.4', 'RARA', 'ADAM19', 'MAFK', 'RNF144B', 'DACT1.1',
            'ARID5B.1', 'RHOB.2', 'MLXIP.1', 'AMIGO2.1', 'GADD45B.1', 'VEGFA.2', 'ADM.2',
            'TGFBI', 'IL11.2', 'DLC1.2', 'VEGFA.5', 'TXNIP.4', 'GCNT1', 'SOCS2.1', 'HMOX1.1',
            'IRS1', 'SKI', 'LMCD1.2', 'CTGF.2', 'JUNB', 'FZD7', 'CTGF', 'FOS.2', 'TOB1', 'RBMS1.1',
            'TIMP1', 'RGS4.1', 'CTPS1.3', 'CEBPD', 'BMP4', 'LTBP2', 'STK38L.2', 'VEGFA.6', 'ADAMTS1',
            'CLCF1', 'KRTAP1-5', 'SH2B3', 'MFSD12', 'LTBP2.1', 'TNFAIP3', 'TXNIP.2', 'SPRY4',
            'SIK1', 'LMCD1.1', 'NR4A1.1', 'EGR1.4', 'PCDH18', 'ANGPTL4', 'PRDM1', 'DUSP1.1',
            'NIPAL4.1', 'UBASH3B', 'TGFBI.3', 'KDM6B.1', 'SMAD7.1', 'HOXA13', 'RCL1', 'GAL',
            'SOX4.1', 'TMEM204', 'TUFT1.2', 'CCDC71L', 'ZNF697', 'SMOX.2', 'TSPAN2.1', 'HAS2',
            'CNKSR3', 'ENC1.1', 'GEM', 'KLHL21', 'CTPS1.2', 'NOL4L', 'RHOB.4', 'ENPP1', 'TPM1',
            'SMAD7.2', 'DUSP10', 'CDKN1B', 'ID1.2', 'FOS', 'ZFP36', 'MEX3B', 'HLX', 'SERTAD2',
            'TFAP2C', 'CTPS1.1', 'GATA2', 'STC2.1', 'BDNF.1', 'MLXIP', 'PRICKLE2', 'RYBP.1',
            'ARID5B.2', 'KDM6B', 'PRICKLE1.1', 'CCNG2.1'}


features = list(set([i.split('.')[0] for i in features]))


class GOTests(unittest.TestCase):
    def setUp(self):
        self.features = features

    def test(self):
        g = GO(self.features[:10])


















