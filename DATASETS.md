# Dataset preparation

This repository does not redistribute the original Numenta Anomaly Benchmark (NAB), Skoltech Anomaly Benchmark (SKAB), Server Machine Dataset (SMD), Soil Moisture Active Passive (SMAP), or Mars Science Laboratory (MSL) benchmark data. Prepare them locally and pass the parent directory with `--data-root`.

## Official download locations

| Benchmark | Upstream location                                                                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NAB       | https://github.com/numenta/NAB                                                                                                                                                             |
| SKAB      | https://github.com/waico/SKAB and https://www.kaggle.com/dsv/1693952                                                                                                                       |
| SMD       | https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset                                                                                                                |
| SMAP      | https://github.com/khundman/telemanom,`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`, and `https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv` |
| MSL       | https://github.com/khundman/telemanom,`https://s3-us-west-2.amazonaws.com/telemanom/data.zip`, and `https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv` |

Check each upstream source for its current license and terms before downloading or redistributing data.

## Expected local layout

```text
<DATA_ROOT>/NAB/raw/labels/combined_windows.json
<DATA_ROOT>/NAB/raw/real*/**/*.csv
<DATA_ROOT>/SKAB/raw/**/*.csv
<DATA_ROOT>/SMD/raw/ServerMachineDataset/test/*.txt
<DATA_ROOT>/SMD/raw/ServerMachineDataset/test_label/*.txt
<DATA_ROOT>/NASA/raw/labeled_anomalies.csv
<DATA_ROOT>/NASA/raw/test/*.npy
```

SMAP and MSL are loaded from the NASA/Telemanom test arrays and `labeled_anomalies.csv`. SMD uses the labeled test arrays. NAB labels are expanded from anomaly windows. SKAB uses numeric columns as features and the `anomaly` column as the binary label.

## Conversion command

From the repository root:

```bash
python scripts/prepare_datasets.py --data-root <DATA_ROOT>
```

The command writes `results/raw/dataset_inventory.csv`. During table generation, `scripts/build_tables.py` derives `manifests/datasets.csv` from this inventory for Table 1 and release verification.

## Inclusion criteria

- observations and binary anomaly labels are aligned;
- at least one contiguous anomaly segment exists;
- original temporal order is preserved.

## Expected family counts

| Benchmark | Series | Median length | Features | Median anomaly rate |
| --------- | -----: | ------------: | -------: | ------------------: |
| NAB       |      7 |          7267 |        1 |               0.100 |
| SKAB      |     34 |          1141 |        8 |               0.350 |
| SMD       |     28 |         23703 |       38 |               0.033 |
| SMAP      |     54 |          8309 |       25 |               0.038 |
| MSL       |     27 |          2277 |       55 |               0.084 |
| Total     |    150 |             - |        - |                   - |

## Preprocessed metadata checksums

These checksums are for the metadata artifacts in this draft, not for redistributed benchmark data.

| File                                  | SHA-256                                                              |
| ------------------------------------- | -------------------------------------------------------------------- |
| `manifests/datasets.csv`            | `272ca4151b8f7d920d9b2cd83406d21dd73e8de84c32d2d0ff51b894824725b8` |
| `results/raw/dataset_inventory.csv` | `c2017b8d2e40d755043fbc31852df129cd50888c211226c6549084b68220a3d9` |

## Included series identifiers

### MSL

msl_C-1, msl_C-2, msl_D-14, msl_D-15, msl_D-16, msl_F-4, msl_F-5, msl_F-7, msl_F-8, msl_M-1, msl_M-2, msl_M-3, msl_M-4, msl_M-5, msl_M-6, msl_M-7, msl_P-10, msl_P-11, msl_P-14, msl_P-15, msl_S-2, msl_T-12, msl_T-13, msl_T-4, msl_T-5, msl_T-8, msl_T-9

### NAB

nab_realAWSCloudwatch_ec2_cpu_utilization_24ae8d, nab_realKnownCause_ambient_temperature_system_failure, nab_realKnownCause_cpu_utilization_asg_misconfiguration, nab_realKnownCause_ec2_request_latency_system_failure, nab_realKnownCause_machine_temperature_system_failure, nab_realKnownCause_nyc_taxi, nab_realKnownCause_rogue_agent_key_hold

### SKAB

skab_other_1, skab_other_10, skab_other_11, skab_other_12, skab_other_13, skab_other_14, skab_other_2, skab_other_3, skab_other_4, skab_other_5, skab_other_6, skab_other_7, skab_other_8, skab_other_9, skab_valve1_0, skab_valve1_1, skab_valve1_10, skab_valve1_11, skab_valve1_12, skab_valve1_13, skab_valve1_14, skab_valve1_15, skab_valve1_2, skab_valve1_3, skab_valve1_4, skab_valve1_5, skab_valve1_6, skab_valve1_7, skab_valve1_8, skab_valve1_9, skab_valve2_0, skab_valve2_1, skab_valve2_2, skab_valve2_3

### SMAP

smap_A-1, smap_A-2, smap_A-3, smap_A-4, smap_A-5, smap_A-6, smap_A-7, smap_A-8, smap_A-9, smap_B-1, smap_D-1, smap_D-11, smap_D-12, smap_D-13, smap_D-2, smap_D-3, smap_D-4, smap_D-5, smap_D-6, smap_D-7, smap_D-8, smap_D-9, smap_E-1, smap_E-10, smap_E-11, smap_E-12, smap_E-13, smap_E-2, smap_E-3, smap_E-4, smap_E-5, smap_E-6, smap_E-7, smap_E-8, smap_E-9, smap_F-1, smap_F-2, smap_F-3, smap_G-1, smap_G-2, smap_G-3, smap_G-4, smap_G-6, smap_G-7, smap_P-1, smap_P-2, smap_P-3, smap_P-4, smap_P-7, smap_R-1, smap_S-1, smap_T-1, smap_T-2, smap_T-3

### SMD

smd_machine-1-1, smd_machine-1-2, smd_machine-1-3, smd_machine-1-4, smd_machine-1-5, smd_machine-1-6, smd_machine-1-7, smd_machine-1-8, smd_machine-2-1, smd_machine-2-2, smd_machine-2-3, smd_machine-2-4, smd_machine-2-5, smd_machine-2-6, smd_machine-2-7, smd_machine-2-8, smd_machine-2-9, smd_machine-3-1, smd_machine-3-10, smd_machine-3-11, smd_machine-3-2, smd_machine-3-3, smd_machine-3-4, smd_machine-3-5, smd_machine-3-6, smd_machine-3-7, smd_machine-3-8, smd_machine-3-9

## Why raw data are excluded

The original benchmark data are not owned by this repository. Each benchmark has its own upstream license, citation, and distribution terms. This draft therefore provides loaders, expected file structure, metadata, and reproducibility commands instead of copying raw benchmark files into the release.
