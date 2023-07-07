#include "open3d/Open3D.h"
#include <fstream>

using namespace open3d;

void readInfo(const std::string& info_file, std::vector<std::string>& stamps)
{
    std::string line;
    std::fstream fs;
    fs.open(info_file);
    if(fs.is_open()) {
      while(std::getline(fs, line)) {
        std::stringstream input(line);
        std::vector<std::string> res;
        res.reserve(18);
        std::string result;
        while(input>>result){
          res.emplace_back(result);
        }
        if(res.size() != 18) {
          return ;
          std::cout << "pose file read failed!\n";
        }  
        stamps.emplace_back(res[16]);
      }
    } else {
      std::cout << "file open failed!\n";
    }
}

void DBSCANFilter(
    std::vector<Eigen::Vector3d>& init_semantics,
    std::vector<Eigen::Vector3d>& seeds,
    std::vector<Eigen::Vector3d>& noise)
{
    const double eps = 0.35;
    int min_points = int(init_semantics.size()/4);

    open3d::geometry::PointCloud pointcloud;
    pointcloud.points_ = init_semantics;
    std::vector<int> labels = pointcloud.ClusterDBSCAN(eps, min_points);
    for(size_t idx = 0; idx < labels.size(); ++idx){
        if(labels[idx] == 0) seeds.emplace_back(init_semantics[idx]);
        else noise.emplace_back(init_semantics[idx]);
    }
}

void DBSCANGrow(
    std::vector<Eigen::Vector3d>& iniseeds,
    std::vector<Eigen::Vector3d>& points,
    std::vector<Eigen::Vector3d>& semantic_points,
    std::vector<Eigen::Vector3d>& other_points)
{
    std::vector<Eigen::Vector3d> seedpoints;
    seedpoints.reserve(iniseeds.size()+points.size());
    seedpoints.insert(seedpoints.end(), iniseeds.begin(), iniseeds.end());
    seedpoints.insert(seedpoints.end(), points.begin(), points.end());

    int seed_size = int(iniseeds.size());
    const double eps = 0.15;
    int min_points = int(iniseeds.size()/4);
    open3d::geometry::PointCloud pointcloud;
    pointcloud.points_ = seedpoints;
    std::vector<int> labels = pointcloud.RegionGrowDBSCAN(eps, min_points, seed_size, false);

    for(size_t idx = 0; idx < labels.size(); ++idx) {
        if(labels[idx] == 0) semantic_points.emplace_back(seedpoints[idx]);
        else other_points.emplace_back(seedpoints[idx]);
    }
    std::cout << semantic_points.size() << " " << other_points.size() << " " << iniseeds.size() << " " << points.size() << std::endl;
}

void processSemanticCloud(
    const double& eps,
    std::shared_ptr<open3d::geometry::PointCloud> pointcloud)
{
    std::vector<Eigen::Vector3d> init_semantics;
    std::vector<Eigen::Vector3d> points;
    for(size_t idx = 0; idx < pointcloud->colors_.size(); ++idx) {
        auto color = pointcloud->colors_[idx];
        auto point = pointcloud->points_[idx];
        int label = static_cast<int>(std::round(color.x() * 255.f));
        if(label == 3) init_semantics.emplace_back(point);
        else points.emplace_back(point);
    }
    if(init_semantics.size() == 0) return;
    std::vector<Eigen::Vector3d> seeds;
    std::vector<Eigen::Vector3d> noise;
    DBSCANFilter(init_semantics, seeds, noise);
    points.insert(points.end(), noise.begin(), noise.end());

    if(seeds.size() == 0) return;

    std::vector<Eigen::Vector3d> semantic_points;
    std::vector<Eigen::Vector3d> other_points;
    DBSCANGrow(seeds, points, semantic_points, other_points);
    Eigen::Vector3d color = Eigen::Vector3d(3.f/255.f, 3.f/255.f, 3.f/255.f);
    std::vector<Eigen::Vector3d> semantic_color(semantic_points.size(), color);
    color = Eigen::Vector3d(1.f/255.f, 1.f/255.f, 1.f/255.f);
    std::vector<Eigen::Vector3d> other_color(other_points.size(), color);

    semantic_color.insert(semantic_color.end(), other_color.begin(), other_color.end());
    semantic_points.insert(semantic_points.end(), other_points.begin(), other_points.end());

    assert(semantic_color.size() == semantic_points.size());
    pointcloud->colors_ = semantic_color;
    pointcloud->points_ = semantic_points;
}

// bool getLabels(std::shared_ptr<open3d::geometry::PointCloud> pointcloud)
// {   
//     std::vector<int> labels;
//     labels.resize(pointcloud->colors_.size());
//     for(size_t idx = 0; idx < pointcloud->colors_.size(); ++idx) {
//         auto color = pointcloud->colors_[idx];
//         int label = static_cast<int>(std::round(color.x() * 255.f));
//         if(label != 3) label = 1;
//         labels[idx] = label;
//     }
//     pointcloud->labels_ = labels;
//     return true;
// }

int main(int argc, char* argv[]) {

    std::string root_path = argv[1];
    std::string info_file = argv[2];

    std::vector<std::string> vstamps;
    readInfo(info_file, vstamps);
    for(const auto& stamp : vstamps){
        std::string pcd_file = root_path + "/shadingData/semantic_pcds/" + stamp + ".pcd";
        std::string out_file = root_path + "/shadingData/cluster_pcds/" + stamp + ".pcd";
        auto pointcloud = io::CreatePointCloudFromFile(pcd_file);
        processSemanticCloud(0.35, pointcloud);
        io::WritePointCloud(out_file, *pointcloud);
        std::cout << out_file << std::endl;
    }

    return 0;
}