#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int16MultiArray.h>
#include <subt_rl/State.h>
#include <tf/tf.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// Pcl load and ros
#include <pcl/PolygonMesh.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

using namespace ros;
using namespace pcl;
using namespace std;

class LiDARVoxel
{
private:
  Subscriber sub_lidar;
  Publisher pub_voxel;
  Publisher pub_visual_voxel;
  Publisher pub_state;
  tf::TransformListener listener;
  Eigen::Matrix4f eigenTrans;
  sensor_msgs::PointCloud2 ros_pc;
  sensor_msgs::PointCloud2 ros_voxel_pc;
  std_msgs::Int16MultiArray lidar_voxel;

  PointCloud<PointXYZ>::Ptr lidar;
  PointCloud<PointXYZI>::Ptr voxel_pc;
  VoxelGrid<PointXYZ> voxel;
  PassThrough<PointXYZ> passX;
  PassThrough<PointXYZ> passY;
  PassThrough<PointXYZ> passZ;
  KdTree<PointXYZ>::Ptr kdtree;

  float gcd = 100;
  float max_xy = 6;
  float max_z = 4;
  float leaf_size = 0.25;
  int z_size = int(max_z/leaf_size);
  int xy_size = int(max_xy*2/leaf_size);
  void cb_lidar(const sensor_msgs::PointCloud2ConstPtr &pc);

public:
  LiDARVoxel(NodeHandle nh);
  ~LiDARVoxel();
};

LiDARVoxel::LiDARVoxel(NodeHandle nh)
{
  tf::StampedTransform transform;
  try
  {
    ros::Duration five_seconds(5.0);
    listener.waitForTransform("X1/base_footprint", "X1/front_laser", ros::Time(0), five_seconds);
    listener.lookupTransform("X1/base_footprint", "X1/front_laser", ros::Time(0), transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }

  pcl_ros::transformAsMatrix(transform, eigenTrans);

  lidar.reset(new PointCloud<PointXYZ>);
  voxel_pc.reset(new PointCloud<PointXYZI>);
  voxel.setLeafSize(leaf_size,leaf_size,leaf_size);

  passX.setFilterFieldName("x");
  passX.setFilterLimits(-max_xy, max_xy);

  passY.setFilterFieldName("y");
  passY.setFilterLimits(-max_xy, max_xy);

  passZ.setFilterFieldName("z");
  passZ.setFilterLimits(0, max_z);

  kdtree.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>);

  lidar_voxel.layout.dim = vector<std_msgs::MultiArrayDimension>(3);
  lidar_voxel.layout.dim[0].label = "x";
  lidar_voxel.layout.dim[0].size = xy_size;
  lidar_voxel.layout.dim[0].stride = z_size * xy_size * xy_size;
  lidar_voxel.layout.dim[1].label = "y";
  lidar_voxel.layout.dim[1].size = xy_size;
  lidar_voxel.layout.dim[1].stride = z_size * xy_size;
  lidar_voxel.layout.dim[2].label = "z";
  lidar_voxel.layout.dim[2].size = z_size;
  lidar_voxel.layout.dim[2].stride = z_size;

  pub_state = nh.advertise<subt_rl::State>("/RL/state", 1);
  pub_voxel = nh.advertise<sensor_msgs::PointCloud2>("/X1/voxel", 1);
  pub_visual_voxel = nh.advertise<sensor_msgs::PointCloud2>("/X1/voxel_visual", 1);
  sub_lidar = nh.subscribe("/X1/points", 1, &LiDARVoxel::cb_lidar, this);
}

LiDARVoxel::~LiDARVoxel()
{
}

void LiDARVoxel::cb_lidar(const sensor_msgs::PointCloud2ConstPtr &pc)
{
  pcl_ros::transformPointCloud(eigenTrans, *pc, ros_pc);
  fromROSMsg(ros_pc, *lidar);

  kdtree->setInputCloud(lidar);
  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);
  kdtree->nearestKSearch(pcl::PointXYZ(0, 0, 0), 1, nn_indices, nn_dists);

  float close_d = sqrt(pow(lidar->points[nn_indices[0]].x, 2) + pow(lidar->points[nn_indices[0]].y, 2) +
                       pow(lidar->points[nn_indices[0]].z, 2));

  voxel.setInputCloud(lidar);
  voxel.filter(*lidar);
  passX.setInputCloud(lidar);
  passX.filter(*lidar);
  passY.setInputCloud(lidar);
  passY.filter(*lidar);
  passZ.setInputCloud(lidar);
  passZ.filter(*lidar);  

  lidar_voxel.data = vector<int16_t>(lidar_voxel.layout.dim[0].stride, 0);
  int sx = lidar_voxel.layout.dim[1].stride;
  int sy = lidar_voxel.layout.dim[2].stride;

  for (PointCloud<PointXYZ>::iterator pt = lidar->begin(); pt < lidar->end(); pt++)
  {
    Eigen::Vector3i cord = voxel.getGridCoordinates(pt->x, pt->y, pt->z);
    lidar_voxel.data[(cord(0) + xy_size/2) * sx + (cord(1) + xy_size/2) * sy + cord(2)] = 1;
  }

  subt_rl::State state;
  state.header.stamp = ros::Time::now();
  state.observation = lidar_voxel;
  state.closestDistance = close_d;
  pub_state.publish(state);


  // ---------------------------------------------------------------------------
  // visualization -------------------------------------------------------------
  // ---------------------------------------------------------------------------

  toROSMsg(*lidar, ros_pc);
  ros_pc.header.frame_id = "X1/base_footprint";
  ros_pc.header.stamp = ros::Time::now();
  pub_voxel.publish(ros_pc);
  lidar->points.clear();

  for (int i = 0; i < lidar_voxel.layout.dim[0].size; i++)
  {
    for (int j = 0; j < lidar_voxel.layout.dim[1].size; j++)
    {
      for (int k = 0; k < lidar_voxel.layout.dim[2].size; k++)
      {
        float intensity = lidar_voxel.data[i * sx + j * sy + k];
        PointXYZI p(intensity);
        p.x = (i - xy_size/2) * leaf_size;
        p.y = (j - xy_size/2) *leaf_size;
        p.z = k * leaf_size;
        voxel_pc->points.push_back(p);
      }
    }
  }
  toROSMsg(*voxel_pc, ros_voxel_pc);
  ros_voxel_pc.header.frame_id = "X1/base_footprint";
  ros_voxel_pc.header.stamp = ros::Time::now();
  pub_visual_voxel.publish(ros_voxel_pc);
  voxel_pc->points.clear();

}

int main(int argc, char *argv[])
{
  init(argc, argv, "LiDARVoxel");
  ROS_INFO("LiDARVoxel init");

  NodeHandle nh;
  LiDARVoxel voxel(nh);
  spin();
  return 0;
}
