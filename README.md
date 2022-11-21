# Point Cloud and Sensor Fusion Materials

## 3D Point Cloud Ground Segmentation with code:
Collect paper about ground segmentation in 3D point cloud. This is the first and a crucial step towards object detection of 3d Point Clouds.

### Geometry and ML Based With Code:
- Fast Segmentation of 3D Point Clouds for Ground Vehicles (2010) [[paper](http://ieeexplore.ieee.org/document/5548059/)], [[code](https://github.com/lorenwel/linefit_ground_segmentation)], [[3rd party implementation](https://github.com/KennyWGH/efficient_online_segmentation)]

- (GP-INASC) On the Segmentation of 3D LIDAR Point Clouds (2011) [[paper](http://ieeexplore.ieee.org/document/5979818/)] [[code](https://github.com/alualu628628/Gaussian-Process-Incremental-Sample-Consensus-GP-INASC)]

- Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications (2017) [[paper](http://ieeexplore.ieee.org/document/7989591/)] [[single plane code](https://github.com/AbangLZU/plane_fit_ground_filter)], [[multi plane code](https://github.com/wangx1996/LIDAR-Segmentation-Based-on-Range-Image)], [[3rd party implementation](https://github.com/chrise96/3D_Ground_Segmentation)]

- A Slope-robust Cascaded Ground Segmentation in 3D Point Cloud for Autonomous Vehicles (2018) [[paper](https://ieeexplore.ieee.org/document/8569534)], [[Python](https://bitbucket.org/n-patiphon/slope_robust_ground_seg)], [[c++](https://github.com/wangx1996/Cascaded-Lidar-Ground-Segmentation)]

- A Probability **Occupancy Grid** Based Approach for Real-Time LiDAR Ground Segmentation (2019) [[paper](https://ieeexplore.ieee.org/document/8666170/)], [[3rd party implementation](https://github.com/MukhlasAdib/KITTI_Mapping)]

- Patchwork: Concentric Zone-based Region-wise Ground Segmentation with Ground Likelihood Estimation Using a 3D LiDAR Sensor (2021) [[paper](https://urserver.kaist.ac.kr/publicdata/patchwork/RA_L_21_patchwork_final_submission.pdf)], [[code](https://github.com/LimHyungTae/patchwork)]

- (RECM-JCP) Fast Ground Segmentation for 3D LiDAR Point Cloud Based on Jump-Convolution-Process (2021) [[paper](https://www.mdpi.com/2072-4292/13/16/3239/xml)], [[code](https://github.com/wangx1996/Fast-Ground-Segmentation-Based-on-JPC)]

- Patchwork++: Fast and Robust Ground Segmentation Solving Partial Under-Segmentation Using 3D Point Cloud (2022) [[paper](https://arxiv.org/pdf/2207.11919.pdf)], [[code](https://github.com/url-kaist/patchwork-plusplus)]

- GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles [[code](https://github.com/anshulpaigwar/GndNet)]

### Elevation and terrain mapping for Robotics:
The elevation mapping technique which was introduced in 2005 during the DARPA challenge by [[Researchers at Stanford University](https://onlinelibrary.wiley.com/doi/10.1002/rob.20147)] was proposed for easier representation of point clouds for ground segementation and real-time autonomous driving. Although, some of the above indicated in literature the use of an elevation method [[RECM-JCP](https://www.mdpi.com/2072-4292/13/16/3239/xml)], a review of thier code shows the use of the polar grid mapping method which is also a very popular and efficient technique for point cloud representation. This section would also include elevation and terrain mapping applied to **SLAM and robotic navigation**. 

- Probabilistic Terrain Mapping for Mobile Robots With Uncertain Localization (2018) [[paper](https://ieeexplore.ieee.org/document/8392399)], [[code](https://github.com/ANYbotics/elevation_mapping)]

- GEM: Online Globally Consistent Dense Elevation Mapping for Unstructured Terrain [[paper](https://ieeexplore.ieee.org/document/9293017)], [[code](https://github.com/ZJU-Robotics-Lab/GEM)] 

- Elevation Mapping for Locomotion and Navigation using GPU [[paper](https://arxiv.org/pdf/2204.12876v1.pdf)], [[code](https://github.com/leggedrobotics/elevation_mapping_cupy)]

- RING++: Roto-translation Invariant Gram for Global Localization on a Sparse Scan Map [[ring](https://arxiv.org/pdf/2204.07992v1.pdf)] [[ring++](https://arxiv.org/pdf/2210.05984v1.pdf)], [[code](https://github.com/MaverickPeter/MR_SLAM)]

- Reconstructing occluded Elevation Information in Terrain Maps with Self-supervised Learning (2022) [[paper](https://ieeexplore.ieee.org/document/9676411)], [[code](https://github.com/mstoelzle/solving-occlusion)]

- Terrain mapping algorithm for motion planning and control by [[robot locomotion](https://github.com/robot-locomotion/terrain-server)]

- [[paper]()], [[code]()]

#### Metrics
- Ground segmentation benchmark in SemanticKITTI dataset by [[url-kaist team](https://github.com/url-kaist/Ground-Segmentation-Benchmark)]

### 2. Clustering Algorithms and Techniques
- A Technical Survey and Evaluation of Traditional Point Cloud Clustering Methods for LiDAR Panoptic Segmentation (2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021W/TradiCV/papers/Zhao_A_Technical_Survey_and_Evaluation_of_Traditional_Point_Cloud_Clustering_ICCVW_2021_paper.pdf)], [[code](https://github.com/placeforyiming/ICCVW21-LiDAR-Panoptic-Segmentation-TradiCV-Survey-of-Point-Cloud-Cluster)]

- FEC: Fast Euclidean Clustering for Point Cloud Segmentation [[paper](https://www.mdpi.com/2504-446X/6/11/325)], [[code](https://github.com/unageek/fast-euclidean-clustering)]

## 2. Deep Learning Segmentation and Object detection Algorithms
- [PointNet](https://stanford.edu/~rqi/pointnet/): Deep Learning on Point Sets for 3D Classification and Segmentation. [[paper](http://arxiv.org/abs/1612.00593)] [[code](https://github.com/charlesq34/pointnet)]
- [PointNet++](https://stanford.edu/~rqi/pointnet2/): Deep Hierarchical Feature Learning on Point Sets in a Metric Space. [[paper](https://arxiv.org/abs/1706.02413)] [[code](https://github.com/charlesq34/pointnet2)]
- Frustum PointNets for 3D Object Detection from RGB-D Data [[paper](https://arxiv.org/pdf/1711.08488.pdf)] [[code](https://github.com/charlesq34/frustum-pointnets)]
- RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving" (ECCV 2020) [[paper](https://arxiv.org/pdf/2001.03343.pdf)] [[code](https://github.com/Banconxuan/RTM3D)] [[code](https://github.com/maudzung/RTM3D)]
- Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds [[paper](https://arxiv.org/pdf/1803.06199.pdf)] [[code](https://github.com/maudzung/Complex-YOLOv4-Pytorch)]
- Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds (The PyTorch implementation) [[code](https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection)]
- VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection [[paper](https://arxiv.org/pdf/1711.06396.pdf)] [[code](https://github.com/qianguih/voxelnet)] [[code](https://github.com/steph1793/Voxelnet)] [[code](https://github.com/TUMFTM/RadarVoxelFusionNet)] [[code](https://github.com/ZhihaoZhu/PointNet-Implementation-Tensorflow)] [[code](https://github.com/jediofgever/PointNet_Custom_Object_Detection)]
- Multi-View 3D Object Detection Network for Autonomous Driving [[paper](https://arxiv.org/pdf/1611.07759)] [[code](https://github.com/bostondiditeam/MV3D)]
- Lightweight and Accurate Point Cloud Clustering. [[paper](https://link.springer.com/article/10.1007/s10514-019-09883-y)] [[code](https://github.com/yzrobot/adaptive_clustering)]
- Linked Dynamic Graph CNN: Learning through Point Cloud by Linking Hierarchical Features. [[paper](https://arxiv.org/pdf/1904.10014.pdf)] [[code](https://github.com/KuangenZhang/ldgcnn)]
- Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Cylindrical_and_Asymmetrical_3D_Convolution_Networks_for_LiDAR_Segmentation_CVPR_2021_paper.pdf)]  [[paper](https://arxiv.org/pdf/2206.02099.pdf)] [[code](https://github.com/cardwing/GitHubs-for-PVKD)]
- Point Transformer. [[paper](https://ieeexplore.ieee.org/document/9552005)] [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)] [[code](https://github.com/engelnico/point-transformer)] [[code](https://github.com/POSTECH-CVLab/point-transformer)]
- GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles. [[paper](https://hal.inria.fr/hal-02927350/document)] [[code](https://github.com/anshulpaigwar/GndNet)]
- DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds. [[paper](https://arxiv.org/pdf/2111.08799.pdf)] [[code](https://github.com/rubenwiersma/deltaconv)]
- PointSIFT: A SIFT-like Network Module for 3D Point Cloud Semantic Segmentation. [[paper](https://arxiv.org/pdf/1807.00652.pdf)] [[code](https://github.com/MVIG-SJTU/pointSIFT)]
- RangeNet++: Fast and Accurate LiDAR Semantic Segmentation. [[paper](https://github.com/LongruiDong/lidar-bonnetal)] [[code](https://github.com/LongruiDong/lidar-bonnetal)]
- Learning Geometry-Disentangled Representation for Complementary Understanding of 3D Object Point Cloud. [[paper](https://arxiv.org/pdf/2012.10921.pdf)] [[code](https://github.com/mutianxu/GDANet)]
- Dynamic Graph CNN for Learning on Point Clouds. [[paper](https://arxiv.org/pdf/1801.07829) [[code](https://github.com/WangYueFt/dgcnn)]
- PointConv: Deep Convolutional Networks on 3D Point Clouds. [[paper](https://arxiv.org/pdf/1811.07246)] [[code](https://github.com/DylanWusee/pointconv)]
- PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. [[paper](https://arxiv.org/pdf/1903.05711.pdf)] [[code](https://github.com/hmgoforth/PointNetLK)]
- PCN: Point Completion Network. [[paper](https://arxiv.org/pdf/1808.00671.pdf)] [[code](https://wentaoyuan.github.io/pcn)]
- RPM-Net: Robust Point Matching using Learned Features. [[paper](https://arxiv.org/pdf/2003.13479.pdf)] [[code](https://github.com/yewzijian/RPMNet)]
- 3D ShapeNets: A Deep Representation for Volumetric Shapes. [[paper](https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf)] [[code](https://github.com/zhirongw/3DShapeNets)]
- Correspondence Matrices are Underrated. [[paper](https://arxiv.org/pdf/2010.16085.pdf)] [[code](https://github.com/tzodge/PCR-CMU)]
- MaskNet: A Fully-Convolutional Network to Estimate Inlier Points. [[paper](https://arxiv.org/pdf/2010.09185.pdf)] [[code](https://github.com/vinits5/masknet)]
- 3DLineDetection. [[paper](https://arxiv.org/pdf/1901.02532.pdf)] [[code](https://github.com/xiaohulugo/3DLineDetection)]
- Deep Learning for 3D Point Clouds: A Survey (IEEE TPAMI, 2020). [[paper](https://arxiv.org/pdf/1912.12033v2.pdf)] [[code](https://github.com/The-Learning-And-Vision-Atelier-LAVA/SoTA-Point-Cloud)]
- LatticeNet: Fast Point Cloud Segmentation Using Permutohedral Lattices. [[paper](https://www.ais.uni-bonn.de/videos/RSS_2020_Rosu/RSS_2020_Rosu.pdf)] [[code](https://github.com/AIS-Bonn/lattice_net)]
- SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation. [[paper](https://arxiv.org/pdf/2004.01803v2.pdf)] [[code](https://github.com/chenfengxu714/SqueezeSegV3)]
- Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds. [[paper](https://arxiv.org/pdf/1806.01759v2.pdf)] [[code](https://github.com/viscom-ulm/MCCNN)]
- cilantro: A Lean, Versatile, and Efficient Library for Point Cloud Data Processing.
- Oriented Point Sampling for Plane Detection in Unorganized Point Clouds. [[paper]()] [[code]()]
- Supervoxel for 3D point clouds. [[paper](https://www.researchgate.net/publication/325334638_Toward_better_boundary_preserved_supervoxel_segmentation_for_3D_point_clouds)] [[code](https://github.com/yblin/Supervoxel-for-3D-point-clouds)]
- MmWave Radar Point Cloud Segmentation using GMM in Multimodal Traffic Monitoring. [[paper](https://arxiv.org/pdf/1911.06364v3.pdf)] [[code](https://github.com/radar-lab/traffic_monitoring)]
- LRGNet: Learnable Region Growing for Class-Agnostic Point Cloud Segmentation. [[paper](https://arxiv.org/pdf/2103.09160v1.pdf)] [[code](https://github.com/jingdao/learn_region_grow)]
- PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds. [[paper](https://arxiv.org/pdf/2103.14635v2.pdf)] [[code](https://github.com/CVMI-Lab/PAConv)]
- SCF-Net: Learning Spatial Contextual Features for Large-Scale Point Cloud Segmentation. [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_SCF-Net_Learning_Spatial_Contextual_Features_for_Large-Scale_Point_Cloud_Segmentation_CVPR_2021_paper.pdf)] [[code](https://github.com/leofansq/SCF-Net)]
- Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling. [[paper](https://arxiv.org/pdf/2111.14819v2.pdf)] [[code](https://github.com/lulutang0608/Point-BERT)]
- RETHINKING NETWORK DESIGN AND LOCAL GEOMETRY IN POINT CLOUD: A SIMPLE RESIDUAL MLP FRAMEWORK. [[paper](https://arxiv.org/pdf/2202.07123v1.pdf)] [[code](https://github.com/ma-xu/pointmlp-pytorch)]
- Masked AutoenGitHubrs for Point Cloud Self-supervised Learning. [[paper](https://arxiv.org/pdf/2203.06604v2.pdf)] [[code](https://github.com/Pang-Yatian/Point-MAE)]
- EagerMOT: 3D Multi-Object Tracking via Sensor Fusion [[paper](https://arxiv.org/pdf/2104.14682.pdf)] [[code](https://github.com/aleksandrkim61/EagerMOT)]
- PointPainting: Sequential Fusion for 3D Object Detection [[paper](https://arxiv.org/pdf/1911.10150.pdf)] [[code](https://github.com/Song-Jingyu/PointPainting)] [[code](https://github.com/AmrElsersy/PointPainting)]
- Automatic Radar-Camera Dataset Generation for Sensor-Fusion Applications [[paper](https://repository.arizona.edu/bitstream/handle/10150/663389/AutoRadarCamera.pdf?sequence=1)] [[code](https://github.com/radar-lab/autolabelling_radar)]
- Radar Voxel Fusion for 3D Object Detection [[paper](https://arxiv.org/pdf/2106.14087.pdf)] [[code](https://github.com/TUMFTM/RadarVoxelFusionNet)]
- CRF-Net for Object Detection (Camera and Radar Fusion Network) [[paper](https://arxiv.org/pdf/2005.07431.pdf)] [[code](https://github.com/TUMFTM/CameraRadarFusionNet)]
- [[paper]()] [[code]()]

### Others
- LCCNet: LiDAR and Camera Self-Calibration using Cost Volume Network [[paper](https://openaccess.thecvf.com/content/CVPR2021W/WAD/papers/Lv_LCCNet_LiDAR_and_Camera_Self-Calibration_Using_Cost_Volume_Network_CVPRW_2021_paper.pdf)] [[code](https://github.com/LvXudong-HIT/LCCNet)]
- A Collection of LiDAR-Camera-Calibration Papers, Toolboxes and Notes [[code](https://github.com/Deephome/Awesome-LiDAR-Camera-Calibration)]
- LiDAR-camera system extrinsic calibration by establishing virtual point correspondences from pseudo calibration objects [[paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-28-12-18261&id=432360)]
- A list of papers and datasets about point cloud analysis (processing) [[code](https://github.com/Yochengliu/awesome-point-cloud-analysis)]
- ICCV-2021-point-cloud-analysis [[code](https://github.com/cuge1995/ICCV-2021-point-cloud-analysis)]
- Lidar and radar fusion for real-time road-objects detection and tracking [[paper](https://www.researchgate.net/publication/351860734_Lidar_and_radar_fusion_for_real-time_road-objects_detection_and_tracking)]
- Awesome Radar Perception [[code](https://github.com/ZHOUYI1023/awesome-radar-perception)]
- Automatic Extrinsic Calibration of Vision and Lidar by Maximizing Mutual Information [[paper](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/112212/rob21542.pdf?sequence=1)]
- Surrounding Objects Detection and Tracking for Autonomous Driving Using LiDAR and Radar Fusion [[paper](https://cjme.springeropen.com/articles/10.1186/s10033-021-00630-y)]
- Camera-LiDAR Multi-Level Sensor Fusion for Target Detection at the Network Edge [[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8227618/)]
- Real-time RADAR and LIDAR Sensor Fusion for Automated Driving [[paper](https://link.springer.com/chapter/10.1007/978-981-15-1366-4_11)]
- Radar Camera Fusion via Representation Learning in Autonomous Driving [[paper](https://openaccess.thecvf.com/content/CVPR2021W/MULA/papers/Dong_Radar_Camera_Fusion_via_Representation_Learning_in_Autonomous_Driving_CVPRW_2021_paper.pdf)]

# REFERENCES
- Paper with Code [[site](https://paperswithcode.com/)]
- 3D point cloud by [[zhulf0804](https://github.com/zhulf0804/3D-PointCloud)]
- Lidar-Ground-Segmantation-Paper-List by [[wangx1996](https://github.com/wangx1996/Lidar-Ground-Segmantation-Paper-List)]