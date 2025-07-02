def get_ukb_serology_field_dict():
    
    ukb_serology_field_dict = {
        # --- Seropositivity Definitions ---
        # These fields typically represent a binary (Yes/No) status based on cutoffs applied to the antigen data.
        'HSV_1_Seropositivity': '30219-0.0',
        'HSV_2_Seropositivity': '30225-0.0',
        'VZV_Seropositivity': '30244-0.0', # Note: VZV seropositivity is often directly from the single gE/gI antigen field
        'EBV_Seropositivity': '30214-0.0',
        'CMV_Seropositivity': '30208-0.0',
        'HHV_6_overall_Seropositivity': '30227-0.0',
        'HHV_6A_Seropositivity': '30228-0.0',
        'HHV_6B_Seropositivity': '30229-0.0',
        'HHV_7_Seropositivity': '30231-0.0',
        'KSHV_Seropositivity': '30237-0.0',
        'HBV_Seropositivity': '30249-0.0',
        'HCV_Seropositivity': '30255-0.0',
        'T_gondii_Seropositivity': '30302-0.0',
        'HTLV_1_Seropositivity': '30299-0.0',
        'HIV_1_Seropositivity': '30295-0.0',
        'BKV_Seropositivity': '30202-0.0',
        'JCV_Seropositivity': '30232-0.0',
        'MCV_Seropositivity': '30238-0.0',
        'HPV_16_Definition_I_Seropositivity': '30263-0.0',
        'HPV_16_Definition_II_Seropositivity': '30264-0.0',
        'HPV_18_Seropositivity': '30269-0.0',
        'C_trachomatis_Definition_I_Seropositivity': '30278-0.0',
        'C_trachomatis_Definition_II_Seropositivity': '30279-0.0',
        'H_pylori_Definition_I_Seropositivity': '30290-0.0',
        'H_pylori_Definition_II_Seropositivity': '30291-0.0',

        # --- Antigen IgG Antibody Measurements (Continuous Values) ---

        # Herpesviruses
        'HSV_1_Antigen_1gG': '30224-0.0',
        'HSV_2_Antigen_2mgG_unique': '30222-0.0',
        'VZV_Antigen_gE_gI': '30245-0.0', # Note: This field is often used for VZV seropositivity directly
        'EBV_Antigen_VCA_p18': '30215-0.0',
        'EBV_Antigen_EBNA_1': '30216-0.0',
        'EBV_Antigen_ZEBRA': '30217-0.0',
        'EBV_Antigen_EA_D': '30218-0.0',
        'CMV_Antigen_pp150_Nter': '30209-0.0',
        'CMV_Antigen_pp_52': '30210-0.0',
        'CMV_Antigen_pp_28': '30211-0.0',
        'HHV_6_Antigen_IE1A': '30230-0.0',
        'HHV_6_Antigen_IE1B': '30233-0.0',
        'HHV_6_Antigen_p101_k': '30234-0.0',
        'HHV_7_Antigen_U14': '30235-0.0',
        'KSHV_Antigen_LANA': '30236-0.0',
        'KSHV_Antigen_K8_1': '30239-0.0',
        'MCV_Antigen_MC_VP1': '30240-0.0', # Merkel Cell Polyomavirus
        'BKV_Antigen_BK_VP1': '30203-0.0', # BK Polyomavirus
        'JCV_Antigen_JC_VP1': '30204-0.0', # JC Polyomavirus

        # Hepatitis Viruses
        'HBV_Antigen_HBc': '30250-0.0',
        'HBV_Antigen_HBe': '30251-0.0',
        'HCV_Antigen_Core': '30252-0.0',
        'HCV_Antigen_NS3': '30253-0.0',

        # Other Pathogens
        'T_gondii_Antigen_sag1': '30303-0.0',
        'T_gondii_Antigen_p22': '30304-0.0',
        'HTLV_1_Antigen_gag': '30300-0.0',
        'HTLV_1_Antigen_env': '30301-0.0',
        'HIV_1_Antigen_gag': '30297-0.0',
        'HIV_1_Antigen_env': '30298-0.0',

        # Human Papillomaviruses (HPV)
        'HPV_16_Antigen_L1': '30265-0.0',
        'HPV_16_Antigen_E6': '30266-0.0',
        'HPV_16_Antigen_E7': '30267-0.0',
        'HPV_18_Antigen_L1': '30270-0.0',

        # Chlamydia trachomatis
        'C_trachomatis_Antigen_momp_D': '30272-0.0',
        'C_trachomatis_Antigen_momp_A': '30273-0.0',
        'C_trachomatis_Antigen_tarp_D_F1': '30274-0.0',
        'C_trachomatis_Antigen_tarp_D_F2': '30275-0.0',
        'C_trachomatis_Antigen_PorB': '30276-0.0',
        'C_trachomatis_Antigen_pGP3': '30277-0.0',

        # Helicobacter pylori
        'H_pylori_Antigen_CagA': '30282-0.0',
        'H_pylori_Antigen_VacA': '30283-0.0',
        'H_pylori_Antigen_OMP': '30284-0.0',
        'H_pylori_Antigen_GroEL': '30285-0.0',
        'H_pylori_Antigen_Catalase': '30286-0.0',
        'H_pylori_Antigen_UreA': '30287-0.0'
    }
    
    return ukb_serology_field_dict
#