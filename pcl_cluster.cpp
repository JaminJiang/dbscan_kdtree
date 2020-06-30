#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/common/common.h>

#include <time.h>
#include <iostream>

#include "DBSCAN_simple.h"
#include "DBSCAN_precomp.h"
#include "DBSCAN_kdtree.h"

// Visualization, [The CloudViewer](https://pcl.readthedocs.io/projects/tutorials/en/latest/cloud_viewer.html#cloud-viewer)
template <typename PointCloudPtrType>
void show_point_cloud(PointCloudPtrType cloud, std::string display_name) {
  pcl::visualization::CloudViewer viewer(display_name);
  viewer.showCloud(cloud);
  while (!viewer.wasStopped())
  {
  }
}

int main(int argc, char** argv) {
    // IO, [Reading Point Cloud data from PCD files](https://pcl.readthedocs.io/projects/tutorials/en/latest/reading_pcd.html#reading-pcd)
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
    reader.read("table_scene_lms400.pcd", *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl;
    show_point_cloud(cloud, "original point cloud");
    
    // Filtering [Downsampling a PointCloud using a VoxelGrid filter](https://pcl.readthedocs.io/projects/tutorials/en/latest/kdtree_search.html#kdtree-search)
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    int downsampled_points_total_size = cloud_filtered->points.size();
    std::cout << "PointCloud after filtering has: " << downsampled_points_total_size  << " data points." << std::endl;
    show_point_cloud(cloud_filtered, "downsampled point cloud");

    // remove the biggest plane
    // Segmentation, Ransac, [Plane model segmentation](https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html#planar-segmentation)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        std::cout << "Could not estimate a planar anymore." << std::endl;
    } else {
        // Filter, [Extracting indices from a PointCloud](https://pcl.readthedocs.io/projects/tutorials/en/latest/extract_indices.html#extract-indices)
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_plane);
        pcl::PointXYZ min, max;
        pcl::getMinMax3D(*cloud_filtered, min, max);
        double min_z = min.z;
        std::cout << "ground plane size: " << cloud_plane->points.size()  << ", min_z:" << min_z << std::endl;
        show_point_cloud(cloud_plane, "gound plane in point cloud");
        // filter planar
        extract.setNegative(true);
        extract.filter(*cloud_f);
        show_point_cloud(cloud_f, "plane filtered point cloud");
        *cloud_filtered = *cloud_f;
    }

    // KdTree, for more information, please ref [How to use a KdTree to search](https://pcl.readthedocs.io/projects/tutorials/en/latest/kdtree_search.html#kdtree-search)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);
    // Segmentation, [Euclidean Cluster Extraction](https://pcl.readthedocs.io/projects/tutorials/en/latest/cluster_extraction.html#cluster-extraction)
    std::vector<pcl::PointIndices> cluster_indices;
    clock_t start_ms = clock();
    
    // test 1. uncomment the following two lines to test the simple dbscan
    // DBSCANSimpleCluster<pcl::PointXYZ> ec;
    // ec.setCorePointMinPts(20);

    // test 2. uncomment the following two lines to test the precomputed dbscan
    // DBSCANPrecompCluster<pcl::PointXYZ>  ec;
    // ec.setCorePointMinPts(20);

    // test 3. uncomment the following two lines to test the dbscan with Kdtree for accelerating
    DBSCANKdtreeCluster<pcl::PointXYZ> ec;
    ec.setCorePointMinPts(20);

    // test 4. uncomment the following line to test the EuclideanClusterExtraction
    // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance(0.05);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    
    clock_t end_ms = clock();
    std::cout << "cluster time cost:" << double(end_ms - start_ms) / CLOCKS_PER_SEC << " s" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZI>);
    int j = 0;
    // visualization, use indensity to show different color for each cluster.
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++, j++) {
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            pcl::PointXYZI tmp;
            tmp.x = cloud_filtered->points[*pit].x;
            tmp.y = cloud_filtered->points[*pit].y;
            tmp.z = cloud_filtered->points[*pit].z;
            tmp.intensity = j%8;
            cloud_clustered->points.push_back(tmp);
        }
    }
    cloud_clustered->width = cloud_clustered->points.size();
    cloud_clustered->height = 1;
    show_point_cloud(cloud_clustered, "colored clusters of point cloud");
    // IO, [Writing Point Cloud data to PCD files](https://pcl.readthedocs.io/projects/tutorials/en/latest/writing_pcd.html#writing-pcd)
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZI>("cloud_clustered.pcd", *cloud_clustered, false);

    return 0;
}