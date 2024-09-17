```{eval-rst}
.. click:: mitgcm_utils.mkMITgcmEXF:app_click
   :prog: mkMITgcmEXF
   :nested: full
```

Example:

```{termynal}
   $ mkMITgcmEXF from-wrf --lonlatbox 29.8,50.2,9.8,30.2 --geo-em-file geo_em.d01.nc wrfout_d01_
   2024-09-17 09:36:29,732 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: PSFC.bin
   2024-09-17 09:36:29,757 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: Q2.bin
   2024-09-17 09:36:29,786 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: T2.bin
   2024-09-17 09:36:29,814 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: ACLWDNB.bin
   2024-09-17 09:36:29,826 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: ACSWDNB.bin
   2024-09-17 09:36:29,858 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: RAINC.bin
   2024-09-17 09:36:29,869 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: U10.bin
   2024-09-17 09:36:29,886 - mitgcm_utils.mkMITgcmEXF - INFO - Writing: V10.bin

```