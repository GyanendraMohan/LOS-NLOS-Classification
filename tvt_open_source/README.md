# Multi-User MIMO dataset with inter-user channel correlation
Codebase for "Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation", which is published on IEEE Transactions on Vehicular Technology 2021.

We used the Argos platform to measure the mobile channels to a 64-antenna massive MIMO base station in outdoor environments during this measurement. We measured user channels in 4 Line-of-Sight and 5 Non-Line-of-Sight clusters in near-stable environments. In each cluster, we measured >25 locations' channels. In total, the measurement covers > 225 locations. And the measured channel at each location covers all 52 data sub-carriers and up to >300 frames. The measurement is done on 2.4-GHz ISM band.

These datasets were collected by Xu Du and Xing Zhang at the Rice University campus. For any questions about this dataset, please contact the paper authors (xdurice@gmail.com) or the RENEW team. For details of the experiment, please refer to Section III-A in the above paper.

In each cluster, we sample > 25 locations. And the collected channel at each cluster can be used as
- a user randomly moving with >3800/7900 (NLOS/LOS) frames over all sub-carriers
- or 25 user channels at each location. Each location will have ~150/~300 frame over all sub-carriers
Please see `Get started` Section for quick-start.

## Code Structure
  ```
  aoa_estimator.py # for Fig. 5, MUSIC based measurement Angle-of-Arrival estimation.
  numerical.py # for Fig. 7-10, Neumerical Simulation.
  measurement.py # for Fig. 3-4, Ovet-The-Air finding plots.
  demo_access_user_channels.py # example for how to get started to access your channels.
  data
  └── intermediate compute results for faster fig plot fine tuning.
  lib
  ├── argos_lib.py # for raw trace processing
  ├── argos_lts.py # dependencies for estimating channels based on Argos 2
  ├── dsp_lib.py # for signal processing/simulation
  └── ota_lib.py # for OTA measurement setups
  plots
  └── figures generated.
  raw_trace
  └── channel data set collected with Argos platform and used in the paper.
  README.md
  LICENSE
  ```

## Requirement

- > 30 G storage
- >= python 3.7
- >= matplotlib 1.3
- >= numpy 1.8

## Reproduce Paper figures
Run the following command to reproduce the measurement findings figures in `plots/`
```
python3 measurement.py
```

Run the following command to reproduce the numerical findings in `plots/`
```
python3 numerical.py
```

Run the following command to produce the AoA estimation based on our measurement:
```
python3 aoa_estimator.py --cluster_name "LOS 3" --location_idx 18  &&
python3 aoa_estimator.py --cluster_name "LOS 4" --location_idx 17 &&
python3 aoa_estimator.py --cluster_name "NLOS 2" --location_idx 16 &&
python3 aoa_estimator.py --cluster_name "NLOS 2" --location_idx 23 &&
python3 aoa_estimator.py --cluster_name "NLOS 3" --location_idx 18
```
which produce figures in `plots/fig5s/`

## Get started in accessing the measurement for your purpose:
`demo_access_user_channels.py` contains two examples for using the channels.

### Example 1: get the channel of each location in each cluster
`get_all_cluster_measured_channel` will returns `Dict[str, Dict[int, PerLocationChannel]]`.
For example, `get_all_cluster_measured_channel()["NLOS 1"][11]` will return the measured wideband channel of user at Location 11 in NLOS 1 cluster. And the measured wideband channel
at each location `PerLocationChannel` is of shape `(NUM_BASE_STATION_ANTENNAS, FRMAE_NUM, SUB_CARRIRE_NUM)`. For example, Location 11 in NLOS 1 cluster has `(64, 226, 52)`, which are 226 channels from users to 64 base-station antennas across all 52 data sub-carriers.

### Example 2: get the channel of a user randomly moving in each cluster
Since we measured channels in each cluster continuously, you can also use the measurement data to get a user moving randomly inside each cluster. `get_single_cluster_channel_mobile` returns a dictionary that maps the cluster name to the channels of a user randonly moving inside the cluster. By running example below, you will see

- User at Cluster LOS 1 has channel (64, 7939, 52)
- User at Cluster LOS 2 has channel (64, 7958, 52)
...
- User at Cluster NLOS 4 has channel (64, 3856, 52)
- User at Cluster NLOS 5 has channel (64, 3862, 52)

And `User at Cluster LOS 1 has channel (64, 7939, 52)` means that user that measured channel of user moving in LOS cluster will have 64 base-station antenna, 7939 frame across 52 sub-carriers.

To run the demo, try the following command
```
python3 demo_access_user_channels.py
```

## Reference
If you use the code or dataset from the project, please cite both of the following paper:
```bibtex
  @ARTICLE{9629284,
  author={Du, Xu and Sabharwal, Ashutosh},
  journal={IEEE Transactions on Vehicular Technology},
  title={Massive MIMO Channels with Inter-User Angle Correlation: Open-Access Dataset, Analysis and Measurement-Based Validation},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TVT.2021.3131606}}


  @ARTICLE{8368089,
  author={Zhang, Xing and Zhong, Lin and Sabharwal, Ashutosh},
  journal={IEEE Transactions on Wireless Communications},
  title={Directional Training for FDD Massive MIMO},
  year={2018},
  volume={17},
  number={8},
  pages={5183-5197},
  doi={10.1109/TWC.2018.2838600}}
```
