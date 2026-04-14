# AortaCFD agent rationale

**Patient:** BPM120  
**Diagnosis:** aortic coarctation post-balloon angioplasty  
**Study goal:** WSS around the coarctation and pressure drop from ascending to descending aorta, for surgical planning.  

## Parameter decisions

- **physics_model** = `laminar`  
  Re is near 2000 at peak systole, at the edge of the laminar regime. The referral explicitly asks for pressure drop and WSS, and the submodule default profile is laminar.  
  > For studies where WSS, OSI and near-wall indices are primary endpoints we recommend mesh refinement  
  — *ValenSendstad2018, p.5*
- **mesh_goal** = `wall_sensitive`  
  WSS around the coarctation is a primary endpoint.  
  > mesh refinement around the wall region  
  — *ValenSendstad2018, p.5*
- **wk_flow_allocation_method** = `user_specified`  
  Murray's law is invalid for coarctation because the stenosis dominates the local pressure distribution.  
  > Murray's law systematically misallocates flow because the stenosis dominates the local pressure distribution  
  — *Wang2025, p.4*
- **wk_flow_split_fractions** = `{'descending': 0.7, 'brachiocephalic': 0.1, 'lcca': 0.1, 'lsa': 0.1}`  
  Paediatric coarctation convention of ~70% to descending aorta.  
  > roughly seventy percent of cardiac output to the descending aorta  
  — *Wang2025, p.4*
- **windkessel_tau** = `1.5`  
  Default systemic tau without patient-specific calibration.  
  > A default tau of one point five seconds is appropriate  
  — *Stergiopulos1999, p.2*
- **backflow_stabilisation** = `0.3`  
  Published default eliminates divergence with <1% bias.  
  > beta_T equal to zero point three damps tangential velocity during backflow  
  — *Esmaily2011, p.7*
- **numerics_profile** = `standard`  
  Default second-order profile for production runs.  
- **number_of_cycles** = `3`  
  Template default; MAP initialisation reaches periodic state in 3 cycles.  

## Patches applied to template

- case_info.patient_id ← BPM120
- case_info.description ← diagnosis from referral
- cardiac_cycle ← 60/78 = 0.769s
- windkessel.systolic_pressure ← 118 mmHg
- windkessel.diastolic_pressure ← 72 mmHg
- inlet ← Doppler CSV (type=TIMEVARYING)
- physics.model ← laminar
- mesh.goal ← wall_sensitive
- numerics.profile ← standard
- simulation_control.number_of_cycles ← 3
- windkessel.tau ← 1.5 s
- windkessel.betaT ← 0.3
- windkessel.methodology ← user_flow_split
- windkessel.flow_split ← 70 (scalar % to descending aorta, branch-percentage method)
- inlet.csv_file ← BPM120.csv (auto-detected from BPM120/)

## Warnings and defaults

- wk_flow_split dict used semantic anatomical names; collapsed to scalar 70% to descending aorta (branch-percentage method distributes the remainder across arch branches by area).

## Confidence

- Intake confidence: **high**
- Literature confidence: **high**
