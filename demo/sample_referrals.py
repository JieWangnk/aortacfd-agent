"""Sample clinical referral texts for the demo app."""

REFERRALS = {
    "BPM120 — Paediatric Coarctation (pre-computed results)": (
        "Patient: BPM120\n\n"
        "12-year-old male, known aortic coarctation post-balloon angioplasty at age 3.\n"
        "Referred for routine follow-up hemodynamic assessment.\n\n"
        "Vitals:\n"
        "  Heart rate 78 bpm, sinus rhythm\n"
        "  Right arm BP 118/72 mmHg, leg BP 104/65 mmHg\n"
        "  Echo cardiac output 4.8 L/min\n"
        "  Body surface area 1.32 m²\n\n"
        "Imaging:\n"
        "  Gated CT angiography segmented into inlet (ascending aorta),\n"
        "  three supra-aortic outlets (BCA, LCC, LSA), descending aorta outlet,\n"
        "  and a wall patch. STLs in mm.\n\n"
        "Clinical question:\n"
        "  Assess wall shear stress around the coarctation and pressure drop from\n"
        "  ascending to descending aorta, for surgical planning. Must finish\n"
        "  overnight on ≤32 cores. Use the 3-element Windkessel default; for this\n"
        "  pathological geometry the literature suggests a user-specified flow split\n"
        "  rather than Murray's law."
    ),
    "Adult Marfan — Aortic Root Dilation": (
        "Patient: MRF042\n\n"
        "34-year-old female with Marfan syndrome. Progressive aortic root dilation\n"
        "from 42mm to 47mm over 18 months. Referred for hemodynamic assessment\n"
        "prior to potential Bentall procedure.\n\n"
        "Vitals:\n"
        "  Heart rate 65 bpm, sinus rhythm\n"
        "  BP 125/78 mmHg (on losartan 50mg daily)\n"
        "  Echo cardiac output 5.2 L/min\n"
        "  BSA 1.85 m²\n\n"
        "Imaging:\n"
        "  ECG-gated CTA with 0.6mm slice thickness, segmented aortic root to\n"
        "  descending aorta. Sinuses of Valsalva dilated (47mm). Three arch\n"
        "  branches and descending aorta outlet.\n\n"
        "Clinical question:\n"
        "  Quantify WSS distribution across the dilated root, particularly at the\n"
        "  sinotubular junction. Pressure drop and flow patterns for surgical timing."
    ),
    "Custom (type your own)": "",
}
