#define _CRT_SECURE_NO_WARNINGS
#include <iomanip>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <ctime>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"
#include "Eigen/Dense"


const int NUM_a = 60;

float pi = 3.1416;
bool *SampleModel;
void readPLY(std::vector<std::vector<float>> &model,char *filepath)
{
	std::cout << filepath << std::endl;
	FILE *file = fopen(filepath, "r");
	printf("reading file...\n");
	std::vector<float>p(6);
	while (!feof(file))
	{
		fscanf(file, "%f %f %f %f %f %f", &p[0], &p[1], &p[2], &p[3], &p[4], &p[5]);
		model.push_back(p);
		//std::cout << p[0] << std::endl;
	}
	fclose(file);
	printf("the ply file has been read successfully!\n");
} 

void compute_bounding_box(std::vector<std::vector<float>>model, std::vector<float>&bounding_box) {
	float minx = model[0][0], miny = model[0][1], minz = model[0][2], maxx = model[0][0], maxy = model[0][1], maxz = model[0][2];
	//计算 bounding box
	for (int count = 0; count < model.size(); count++)
	{
		if (model[count][0] > maxx)		maxx = model[count][0];
		else if (model[count][0] < minx)		minx = model[count][0];
		if (model[count][1] > maxy)		maxy = model[count][1];
		else if (model[count][1] < miny)		miny = model[count][1];
		if (model[count][2] > maxz)		maxz = model[count][2];
		else if (model[count][2] < minz)		minz = model[count][2];
	}
	float X = maxx - minx;
	float Y = maxy - miny;
	float Z = maxz - minz;
	float diameter = sqrt(X * X + Y * Y + Z * Z);
	bounding_box.push_back(diameter);
	bounding_box.push_back(X);
	bounding_box.push_back(Y);
	bounding_box.push_back(Z);
	bounding_box.push_back(minx);
	bounding_box.push_back(maxx);
	bounding_box.push_back(miny);
	bounding_box.push_back(maxy);
	bounding_box.push_back(minz);
	bounding_box.push_back(maxz);
}

class Pose3D
{
public:
	//private:
	float alpha;
	float modelIndex;
	float numVotes;
	Eigen::Matrix4f pose;

	//float voters;
	float angle;
	Eigen::Vector3f omega;
	Eigen::Vector4f q;
	Pose3D();
	Pose3D(float a, float b, float c);

	~Pose3D();
	void updatePose(Pose3D &obj, Eigen::Matrix4f newPose);
	void updatePoseT(Pose3D &obj, Eigen::Vector3f t);
	void updatePoseQuat(Pose3D &obj, Eigen::Vector4f q);

	
};

class PPF3DDetector
{
private:
	float samplingRalative;//0.04,  调试的时候用0.1
	float distanceStep;    //采样率*模型直径，即划分的每个小box的直径
	int angleRelative = 30; 
	float angleRadians = (360 / angleRelative)*pi / 180;
	int refPointNum;        //模型采样点数
	float modelDiameter;     
	float sceneDiameter;
	float scenesamplingRelative;
	bool train=false;
	std::vector<std::vector<float>> sampledPC;
	std::map<float, std::vector<std::vector<float>> >  hashTable;   //创建map容器作为哈希表
	std::vector<Pose3D>  poseList;

public:
	PPF3DDetector();
	~PPF3DDetector();
	void samplePCpoisson(std::vector<std::vector<float>>, float,std::vector<std::vector<float>> &);
	void computePPF(std::vector<float> , std::vector<float> , std::vector<float> &);
	void trainModel(std::vector<std::vector<float>>, float);
	void clusterPoses(std::vector<Pose3D>, std::vector<std::vector<Pose3D>> &, std::vector<float>&);
	std::vector<Pose3D> recomputeScore(std::vector<Pose3D> poses, std::vector<std::vector<float>> scene);
	void matchScene(std::vector<Pose3D> &,bool,bool, std::vector<std::vector<float>>, float);
};
PPF3DDetector::PPF3DDetector()
{
}
PPF3DDetector::~PPF3DDetector()
{
}

float compute_distance_square(std::vector<float>p1, std::vector<float>p2) {
	float distance = (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2]);
	return distance;
}

void PPF3DDetector::samplePCpoisson(std::vector<std::vector<float>>model, float sampleStep, std::vector<std::vector<float>> &sampledPoint)  //每个栅格取一个点
{
	printf("sampling pc ...\n");
	struct Index {
		int i;
		int j;
		int k;
	};

	std::vector<float>b_box;
	compute_bounding_box(model, b_box);
	float r = b_box[0] * sampleStep;
	float rs = r * r;
	float boxSize = r / (float)sqrt(3);
	int  samplesInDimX = floor(b_box[1] / boxSize);  //27
	int  samplesInDimY = floor(b_box[2] / boxSize);  //26
	int  samplesInDimZ = floor(b_box[3] / boxSize);  //19
	SampleModel = new bool[model.size()+1];
	std::vector<Index> index(model.size());
	//int ***map;
	//map = new int **[samplesInDimX+2];
	//for (int i = 0; i <= samplesInDimX+1; i++) {
	//	map[i] = new int *[samplesInDimY+2];
	//	for (int j = 0; j <= samplesInDimY+1; j++) {
	//		map[i][j] = new int [samplesInDimZ+2];
	//		for (int k = 0; k <= samplesInDimZ+1; k++) {
	//			map[i][j][k] = 0;
	//			//std::cout << i << '\t' << j << '\t' << k << '\n';
	//		}
	//	}
	//}
	std::vector<std::vector<std::vector<int>>> map;
	for (int i = 0; i <= samplesInDimX+1; i++) {
		std::vector<std::vector<int>> map_j;
		map.push_back(map_j);
		for (int j = 0; j <= samplesInDimY+1; j++) {
			std::vector<int> map_k;
				map[i].push_back(map_k);
			for (int k = 0; k <= samplesInDimZ+1; k++) {
				map[i][j].push_back(0);
			}
		}
	}

	for (int count = 0; count < model.size(); count++) {
		SampleModel[count] = true;
		index[count].i = floor(samplesInDimX*(model[count][0] - b_box[4]) / b_box[1])+1;
		index[count].j = floor(samplesInDimY*(model[count][1] - b_box[6]) / b_box[2])+1;
		index[count].k = floor(samplesInDimZ*(model[count][2] - b_box[8]) / b_box[3])+1;
		
		for (int i = std::max(1,index[count].i - 2); i <= std::min(index[count].i+2, samplesInDimX); i++) {
			for (int j = std::max(1,index[count].j - 2); j <= std::min(index[count].j+2, samplesInDimY); j++) {
				for (int k = std::max(1, index[count].k - 2); k <= std::min(index[count].k+2, samplesInDimZ); k++) {
					int map_index = map[i][j][k];
					if (map_index == 0) {
						continue;
					}
					else{
						float temp_dis = compute_distance_square(model[count], model[map_index]);
						if (temp_dis < rs) {
							SampleModel[count] = false;
							break;
						}
					}
				}

			}
		}
		if (SampleModel[count] == true) {
			map[index[count].i][index[count].j][index[count].k] = count;
		}
		if (count % 25000 == 0)
		{
			std::cout << "sample: " << std::setw(4) << round(100 * count / model.size()) << " %.\r";
		}
	}
	std::cout << "sample: " << std::setw(4) << 100 << " %.\n";
	std::vector<float> temp_p(6);
	for (int  k = 1; k <= samplesInDimZ; k++) {
		for (int j = 1; j <= samplesInDimY; j++) {
			for (int i = 1; i <= samplesInDimX; i++) {
				if (map[i][j][k] != 0) {
					int count = map[i][j][k];
					//采样点，并归一化法矢
					temp_p[0] = model[count][0];
					temp_p[1] = model[count][1];
					temp_p[2] = model[count][2];
					float no = sqrt(model[count][3] * model[count][3] + model[count][4] * model[count][4] + model[count][5] * model[count][5]);
					temp_p[3] = model[count][3] / no;
					temp_p[4] = model[count][4] / no;
					temp_p[5] = model[count][5] / no;
					sampledPoint.push_back(temp_p);
				}
			}
		}
	}

	//for (int i = 0; i < samplesInDimX; i++)
	//{
	//	for (int j = 0; j < samplesInDimY; j++)
	//	{
	//		delete[] map[i][j];
	//	}
	//}
	//for (int i = 0; i < samplesInDimX; i++)
	//{
	//	delete[] map[i];
	//}
	//delete[] map;

	//std::cout << "the sampled pc size:" << sampledPoint.size() << std::endl;

	//以下采用的方法为  每个栅格提取第一个点作为采样点
	//for (int i = 0; i < samplesInDimX; i++) {
	//	for (int j = 0; j < samplesInDimY; j++) {
	//		for (int k = 0; k < samplesInDimZ; k++) {
	//			for (int count = 0; count < model.size(); count++) {
	//				if (SampleModel[count] == false)
	//					if (index[count].i == i && index[count].j == j && index[count].k == k) {
	//						SampleModel[count] = true;
	//						break;
	//					}
	//			}
	//		}
	//	}
	//}
	//std::vector<float> temp_p(6);
	//for (int count = 0; count < model.size(); count++) {
	//	if (SampleModel[count] == true) {
	//		//采样点，并归一化法矢
	//		temp_p[0] = model[count][0];
	//		temp_p[1] = model[count][1];
	//		temp_p[2] = model[count][2];
	//		float no = sqrt(model[count][3] * model[count][3] + model[count][4] * model[count][4] + model[count][5] * model[count][5]);
	//		temp_p[3] = model[count][3] / no;
	//		temp_p[4] = model[count][4] / no;
	//		temp_p[5] = model[count][5] / no;
	//		sampledPoint.push_back(temp_p);
	//	}
	//}
	//std::cout << "the sampled pc size:" << sampledPoint.size() << std::endl;

}

void PPF3DDetector::computePPF(std::vector<float> p1, std::vector<float> p2, std::vector<float> &ppf_value)
{
	Eigen::Vector3f d(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
	float dn = d.norm();
	Eigen::Vector3f n1(p1[3], p1[4], p1[5]);
	Eigen::Vector3f n2(p2[3], p2[4], p2[5]);
	float f1, f2, f3;
	if (dn > 0)
	{
		Eigen::Vector3f dNorm(d(0) / dn, d(1) / dn, d(2) / dn);
		f1 = atan2((dNorm.cross(n1)).norm(), dNorm.dot(n1));
		f2 = atan2((dNorm.cross(n2)).norm(), dNorm.dot(n2));
		f3 = atan2((n1.cross(n2)).norm(), n1.dot(n2));
	}
	else
	{
		f1 = 0;
		f2 = 0;
		f3 = 0;
	}
	ppf_value.push_back(f1);
	ppf_value.push_back(f2);
	ppf_value.push_back(f3);
	ppf_value.push_back(dn);

}

//------------------train--------------------------使用了hash编码，对模型进行训练构造hash Table
uint32_t murmurhash3(const uint32_t *key, uint32_t len, uint32_t seed) {
	static const uint32_t c1 = 0xcc9e2d51;
	static const uint32_t c2 = 0x1b873593;
	static const uint32_t r1 = 15;
	static const uint32_t r2 = 13;
	static const uint32_t m = 5;
	static const uint32_t n = 0xe6546b64;
	uint32_t hash = seed;
	const int nblocks = len / 4;
	const uint32_t *blocks = (const uint32_t *)key;
	/*
	std::cout << "blocks: " << blocks[0] << std::endl;
	std::cout << "blocks: " << blocks[1] << std::endl;
	std::cout << "blocks: " << blocks[2] << std::endl;
	std::cout << "blocks: " << blocks[3] << std::endl;
	std::cout << "blocks: " << (const uint32_t *)blocks[0] << std::endl;
	std::cout << "blocks: " << (const uint32_t *)blocks[1] << std::endl;
	std::cout << "blocks: " << (const uint32_t *)blocks[2] << std::endl;
	std::cout << "blocks: " << (const uint32_t *)blocks[3] << std::endl;
	*/
	//const uint32_t blocks[4] = { 8,6,1,1 };
	int i;
	for (i = 0; i < nblocks; i++) {
		uint32_t k = blocks[i];
		k *= c1;
		k = (k << r1) | (k >> (32 - r1));
		k *= c2;
		hash ^= k;
		hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
	}

	const uint8_t *tail = (const uint8_t *)(key + nblocks * 4);
	uint32_t k1 = 0;

	switch (len & 3) {
	case 3:
		k1 ^= tail[2] << 16;
	case 2:
		k1 ^= tail[1] << 8;
	case 1:
		k1 ^= tail[0];

		k1 *= c1;
		k1 = (k1 << r1) | (k1 >> (32 - r1));
		k1 *= c2;
		hash ^= k1;
	}

	hash ^= len;
	hash ^= (hash >> 16);
	hash *= 0x85ebca6b;
	hash ^= (hash >> 13);
	hash *= 0xc2b2ae35;
	hash ^= (hash >> 16);

	//std::cout << hash << std::endl;

	return hash;
}

uint32_t hashPPF(std::vector<float> ppf_value, float angleRadians, float distanceStep)
{
	const uint32_t key[4] = { int(ppf_value[0] / angleRadians), int(ppf_value[1] / angleRadians) , int(ppf_value[2] / angleRadians), int(ppf_value[3] / distanceStep) };
	int len = 16, seed = 42;
	uint32_t hout = murmurhash3(key, 16, 42);
	return hout;
}

void transformRT(std::vector<float> p, Eigen::Matrix3f &R, Eigen::Vector3f &t)
{
	float angle = acos(p[3]);//旋转角度
	Eigen::Vector3f axis(0, p[5], -p[4]);//旋转轴
	if (p[4] == 0 && p[5] == 0) { axis(0) = 0; axis(1) = 1;  axis(2) = 0;}
	else {
		float axisNorm = axis.norm();
		if (axisNorm > 0) {	
			axis(0) = axis(0) / axisNorm;
			axis(1) = axis(1) / axisNorm;
			axis(2) = axis(2) / axisNorm;
		}
	}
	Eigen::AngleAxisf rotationVector(angle, axis.normalized());
	R = Eigen::Matrix3f::Identity();
	R = rotationVector.toRotationMatrix();//所求旋转矩阵
	Eigen::Vector3f p_temp(p[0], p[1], p[2]);
	Eigen::Vector3f t_temp;
	t_temp = (-1)*R*p_temp;
	t= t_temp;
}

float computeAlpha(std::vector<float> p1, std::vector<float> p2)
{
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	Eigen::Vector3f p2_temp(p2[0], p2[1], p2[2]);
	transformRT(p1,R,t);
	Eigen::Vector3f mpt =  R * p2_temp + t;
	float alpha = atan2(-mpt(2), mpt(1));
	if (sin(alpha)*mpt(2) > 0) { alpha = -alpha; }
	return alpha;
}

void PPF3DDetector::trainModel(std::vector<std::vector<float>>model, float samplingRalative = 0.04)
{
	std::vector<float>b_box;
	compute_bounding_box(model, b_box);
	modelDiameter = b_box[0];
	distanceStep = modelDiameter * samplingRalative;

	//poisson 采样  归一化法矢 采样点组成列向量 作为后续的参考点

	/*泊松采样-------------------------------------*/

	samplePCpoisson(model, samplingRalative, sampledPC);
	////std::vector<std::vector<float>> sampledPC_temp;
	////for (int i = 0; i < 281; i++) {
	////		sampledPC_temp.push_back(sampledPC[i]);
	////}
	////std::vector<float>temp;
	////temp.push_back(-8.1563);
	////temp.push_back(-18.0921);
	////temp.push_back(-340.7340);
	////temp.push_back(-0.0469);
	////temp.push_back(-0.6549);
	////temp.push_back(-0.7543);
	////sampledPC_temp.push_back(temp);
	////for (int i = 281; i < sampledPC.size(); i++) {
	////	sampledPC_temp.push_back(sampledPC[i]);
	////}
	////
	////sampledPC.clear();
	////sampledPC = sampledPC_temp;
	//std::ofstream out1("samplemodel.txt");
	//for (int i = 0; i < sampledPC.size(); i++) {
	//	std::cout << std::setw(10) << sampledPC[i][0] << '\t' << i << std::endl;
	//	out1 << sampledPC[i][0]<< '\n';
	//}
	//out1.close();

	/*泊松采样-------------------------------------*/

	std::cout << "the sampled MODEL pc size:" << sampledPC.size() << std::endl;

	/*----------------------以上采样点少一，手动补上，稍后修改 -----------------*/


	refPointNum = sampledPC.size();
	//hash值用float 存储，  int 位数不足；
	std::map<float, std::vector<std::vector<float>> >  hashTableLoc;   //创建map容器作为哈希表
	float lambda = 0.98;
	for (int i = 0; i < refPointNum; i++){
		std::vector<float>p1 = sampledPC[i];
		for (int j = 0; j < refPointNum; j++){
			if (i != j){
				std::vector<float> f;
				std::vector<float>p2 = sampledPC[j];

				computePPF(p1, p2,f);
				uint32_t hash = hashPPF(f, angleRadians, distanceStep);
				float alphaAngle = computeAlpha(p1, p2);
				float ppfInd = i * refPointNum + (j - 1);
				float dp = sampledPC[i][3] * sampledPC[j][3] + sampledPC[i][4] * sampledPC[j][4] + sampledPC[i][5] * sampledPC[j][5];
				float voteValue = 1 - lambda * std::abs(dp);
				std::vector<float> node(4);
				node[0] = i;
				node[1] = ppfInd;
				node[2] = alphaAngle;
				node[3] = voteValue;
				hashTableLoc[hash].push_back(node);
			}
		}
		if (i % 80 == 0) 
		{ 
			std::cout << "trained: " <<std::setw(4)<< round(100 * i / refPointNum) << " %.\r";
		}
	}
	std::cout << "trained: " << std::setw(4) << 100 << " %.\n";
	hashTable = hashTableLoc;
	train = true;
	std::cout << "Training Finished.\n";
}

Eigen::Matrix4f XrotMat(float angle){
	Eigen::Matrix4f T;
	T << 1, 0, 0, 0, 0, cos(angle), -sin(angle), 0, 0, sin(angle), cos(angle), 0, 0, 0, 0, 1;
	return T;
}


Pose3D::Pose3D()
{
	alpha = 0;
	modelIndex = 0;
	numVotes = 0;
	pose = Eigen::Matrix4f::Zero();
}
Pose3D::Pose3D(float a, float b, float c)
{
	alpha = a;
	modelIndex = b;
	numVotes = c;
	pose = Eigen::Matrix4f::Zero();
}

Pose3D::~Pose3D()
{
}

void dcm2quat(Eigen::Matrix3f R, Eigen::Vector4f &q) {
	float tr = R.trace();
	float nr;
	Eigen::Vector4d q_temp;
	
	if (tr > 0) {
		q_temp << tr + 1, R(1, 2) - R(2, 1), R(2, 0) - R(0, 2), R(0, 1) - R(1, 0);
		nr = q_temp(0);
	}
	else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
		q_temp << R(1, 2) - R(2, 1),1+R(0,0)-R(1,1)-R(2,2) , R(1, 0) + R(0, 1), R(2, 0) + R(0, 2) ;
		nr = q_temp(1);
	}
	else if (R(1, 1) > R(2, 2)) {
		q_temp << R(2, 0) - R(0, 2), R(1, 0) + R(0, 1), 1 + R(1, 1) - R(0, 0) - R(2, 2) , R(2, 1) + R(1, 2);
		nr = q_temp(2);
	}
	else {
		q_temp << R(0, 1) - R(1, 0), R(2, 0) + R(0, 2), R(2, 1) + R(1, 2),1 + R(2, 2) - R(0, 0) - R(1, 1) ;
		nr = q_temp(3);
	}
	double nr_sqrt_coff = (0.5)/std::sqrt(nr);
	q << q_temp(0)*nr_sqrt_coff, q_temp(1)*nr_sqrt_coff, q_temp(2)*nr_sqrt_coff, q_temp(3)*nr_sqrt_coff;
}



void Pose3D::updatePose(Pose3D &obj, Eigen::Matrix4f newPose) {
	obj.pose = newPose;
	Eigen::Matrix3f pose_part;
	pose_part << obj.pose(0, 0), obj.pose(0, 1), obj.pose(0, 2), obj.pose(1, 0), obj.pose(1, 1), obj.pose(1, 2), obj.pose(2, 0), obj.pose(2, 1), obj.pose(2, 2);
	//直接使用fromRotationMatrix() 旋转矩阵   旋转向量；
	Eigen::AngleAxisf  vrot;
	vrot.fromRotationMatrix(pose_part);
	//std::cout << "vrot:\n" <<vrot.axis()[0];
	obj.omega << vrot.axis()[0], vrot.axis()[1], vrot.axis()[2];
	obj.angle = vrot.angle();
	//直接使用旋转矩阵赋值四元数(w,x,y,z)；
	
	//Eigen::Quaternionf q_temp = Eigen::Quaternionf(pose_part);
	dcm2quat(pose_part, obj.q);
}

void Pose3D::updatePoseT(Pose3D &obj, Eigen::Vector3f t) {
	Pose3D obj_temp;
	obj_temp.pose << obj.pose(0, 0), obj.pose(0, 1), obj.pose(0, 2), t(0), obj.pose(1, 0), obj.pose(1, 1), obj.pose(1, 2), t(1), obj.pose(2, 0), obj.pose(2, 1), obj.pose(2, 2), t(2), 0, 0, 0, 1;
	obj.pose = obj_temp.pose;
}

Eigen::Matrix3f quat2dcm(Eigen::Vector4f q) {
	Eigen::Matrix3f dcm = Eigen::Matrix3f::Zero();
	dcm(0, 0) = pow(q(0), 2) + pow(q(1), 2) - pow(q(2), 2) - pow(q(3), 2);
	dcm(0, 1) = 2.0*(q(1)*q(2) + q(0)*q(3));
	dcm(0, 2) = 2.0*(q(1)*q(3) - q(0)*q(2));
	dcm(1, 0) = 2.0*(q(1)*q(2) - q(0)*q(3));
	dcm(1, 1) = pow(q(0), 2) - pow(q(1), 2) + pow(q(2), 2) - pow(q(3), 2);
	dcm(1, 2) = 2.0*(q(2)*q(3) + q(0)*q(1));
	dcm(2, 0) = 2.0*(q(1)*q(3) + q(0)*q(2));
	dcm(2, 1) = 2.0*(q(2)*q(3) - q(0)*q(1));
	dcm(2, 2) = pow(q(0), 2) - pow(q(1), 2) - pow(q(2), 2) + pow(q(3), 2);

	return dcm;
}

void Pose3D::updatePoseQuat(Pose3D &obj, Eigen::Vector4f qNew) {
	obj.q = qNew;
	Eigen::Matrix4f oldPose = obj.pose;
	Eigen::Matrix3f dcm;
	dcm = quat2dcm(qNew);
	Eigen::Matrix4f Newpose;
	Newpose << dcm(0, 0), dcm(0, 1), dcm(0, 2), oldPose(0, 3), dcm(1, 0), dcm(1, 1), dcm(1, 2), oldPose(1, 3), dcm(2, 0), dcm(2, 1), dcm(2, 2), oldPose(2, 3), 0, 0, 0, 1;
	obj.pose = Newpose;

	//angle axis 
	Eigen::Matrix3f pose_part;
	pose_part << obj.pose(0, 0), obj.pose(0, 1), obj.pose(0, 2), obj.pose(1, 0), obj.pose(1, 1), obj.pose(1, 2), obj.pose(2, 0), obj.pose(2, 1), obj.pose(2, 2);
	//直接使用fromRotationMatrix() 旋转矩阵   旋转向量；
	Eigen::AngleAxisf  vrot;
	vrot.fromRotationMatrix(pose_part);
	//std::cout << "vrot:\n" <<vrot.axis()[0];
	obj.omega << vrot.axis()[0], vrot.axis()[1], vrot.axis()[2];
	obj.angle = vrot.angle();


}


//选择法排序
void sortPoses(std::vector<Pose3D>  poseList, std::vector<Pose3D>  &sorted) {
	
	int i, j, p;
	for (i = 0; i < poseList.size()-1; i++) {
		p = i;
		for (j = i + 1; j < poseList.size(); j++)
			if (poseList[j].numVotes > poseList[p].numVotes) p = j;
		if (p != j) {
			Pose3D t = poseList[p];
			poseList[p] = poseList[i];
			poseList[i] = t;
		}
	}
	for (int count = 0; count < poseList.size(); count++) {
		sorted.push_back(poseList[count]);
		//std::cout << std::setw(5) << count << '\t' << sorted[count].numVotes << '\n';
	}
}

bool comparePoses(Pose3D p1, Pose3D p2, float objectDiameter, float angleRadiansLoc) {
	float d = sqrt((p1.pose(0, 3) - p2.pose(0, 3))*(p1.pose(0, 3) - p2.pose(0, 3)) + (p1.pose(1, 3) - p2.pose(1, 3))*(p1.pose(1, 3) - p2.pose(1, 3)) + (p1.pose(2, 3) - p2.pose(2, 3))*(p1.pose(2, 3) - p2.pose(2, 3)));
	float phi = std::abs(p1.angle - p2.angle);
	if (d < (0.1*objectDiameter) && phi < angleRadiansLoc)
	{
		return true;
	}
	else {
		return false;
	}
}

void PPF3DDetector::clusterPoses(std::vector<Pose3D> poseList, std::vector<std::vector<Pose3D>> &cluster, std::vector<float> &votes) {
	float modelDiameterLoc = modelDiameter;
	float angleRadiansLoc = angleRadians;
	std::vector<Pose3D>  sorted;
	sortPoses(poseList, sorted);
	std::vector<std::vector<Pose3D>> poseClusters;
	for (int count = 0; count < sorted.size(); count++) {
		votes.push_back(0);
		std::vector<Pose3D> a;
		poseClusters.push_back(a);
	}
	int clusterNum = 0;
	for (int i = 0; i < sorted.size(); i++)
	{
		bool assigned = false;
		Pose3D curPose = sorted[i];
		for (int j = 0; j < clusterNum; j++) {
			Pose3D poseCenter = poseClusters[j][0];
			if (comparePoses(curPose, poseCenter, modelDiameterLoc, angleRadiansLoc)) {
				poseClusters[j].push_back(curPose);
				votes[j] = votes[j] + curPose.numVotes;
				assigned = true;
				break;
			}
		}
		if (!assigned) {
			poseClusters[clusterNum].push_back(curPose);
			votes[clusterNum ] = curPose.numVotes;
			clusterNum = clusterNum + 1;
		}
	}
	for (int i = 0; i < clusterNum; i++) {
		std::vector<Pose3D> a;
		cluster.push_back(a);
		for (int j = 0; j < poseClusters[i].size(); j++)
			cluster[i].push_back(poseClusters[i][j]);
	}
}

void avg_quaternion_markley(Eigen::Vector4f &qs_avg, std::vector< Eigen::Vector4f> qs) {
	Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
	int M = qs.size();
	for (int i = 0; i < M; i++) {
		Eigen::Vector4f qs_temp = qs[i];
		A = qs_temp*qs_temp.transpose() + A;
	}
	A = ((float)1.0 / (float)M)*A;

	Eigen::EigenSolver<Eigen::Matrix4f> es(A);
	Eigen::MatrixXcf evecs = es.eigenvectors();//获取矩阵特征向量4*4，这里定义的MatrixXcd必须有c，表示获得的是complex复数矩阵
	Eigen::MatrixXcf evals = es.eigenvalues();//获取矩阵特征值 4*1
	Eigen::MatrixXf evalsReal;//注意这里定义的MatrixXd里没有c
	evalsReal = evals.real();//获取特征值实数部分
	Eigen::MatrixXf::Index evalsMax;
	evalsReal.rowwise().sum().maxCoeff(&evalsMax);//得到最大特征值的位置
	Eigen::Vector4f q;
	q << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax), evecs.real()(3, evalsMax);//得到对应特征向量
	qs_avg =q;
}

void averageClusters(std::vector<std::vector<Pose3D>> cluster, std::vector<float> votes, std::vector<Pose3D> &resultPoses){
	for (int i = 0; i < cluster.size(); i++) {
		Pose3D newPose(0, -1, votes[i]);
		std::vector< Eigen::Vector4f> qs;
		std::vector<Eigen::Vector3f> ts;
		for (int count = 0; count < cluster[i].size();count++) {
			qs.push_back(cluster[i][count].q);
			Eigen::Vector3f ts_temp;
			ts_temp << cluster[i][count].pose(0, 3), cluster[i][count].pose(1, 3), cluster[i][count].pose(2, 3);
			ts.push_back(ts_temp);
		}
		Eigen::Vector3f t_mean;
		float t1=0, t2=0, t3=0;
		for (int count = 0; count < ts.size(); count++) {
			t1 = t1+ts[count][0];
			t2 = t2+ts[count][1];
			t3 = t3+ts[count][2];
		}
		t_mean << t1 / (float)ts.size(), t2 / (float)ts.size(), t3 / (float)ts.size();

		newPose.updatePoseT(newPose,t_mean);
		Eigen::Vector4f qs_avg;
		avg_quaternion_markley(qs_avg, qs);
		newPose.updatePoseQuat(newPose,qs_avg);
		
		resultPoses.push_back(newPose);

	}

	
	
}

std::vector<std::vector<float>>  transformPose(std::vector<std::vector<float>>pc, Eigen::Matrix4f pose) {
	std::vector<std::vector<float>> newpc=pc;
	for (int count = 0; count < pc.size(); count++) {
		Eigen::Vector4f p;
		p<<pc[count][0], pc[count][1], pc[count][2],1;
		p = pose * p;
		newpc[count][0] = p(0);
		newpc[count][1] = p(1);
		newpc[count][2] = p(2);
		Eigen::Vector3f n;
		Eigen::Vector3f p_n;
		Eigen::Matrix3f p_M;
		p_M << pose(0, 0), pose(0, 1), pose(0, 2), pose(1, 0), pose(1, 1), pose(1, 2), pose(2, 0), pose(2, 1), pose(2, 2);
		p_n << pc[count][3], pc[count][4], pc[count][5];
		n = p_M * p_n;
		float nNorm = sqrt(n(0)*n(0) + n(1)*n(1) + n(2)*n(2));
		if (nNorm > 0) {
			n(0) = n(0) / nNorm;
			n(1) = n(1) / nNorm;
			n(2) = n(2) / nNorm;
			newpc[count][3] = n(0);
			newpc[count][4] = n(1);
			newpc[count][5] = n(2);
		}
	}
	return newpc;
}

std::vector<Pose3D> PPF3DDetector::recomputeScore(std::vector<Pose3D> poses, std::vector<std::vector<float>> scene) {
	std::vector<Pose3D> outPoses= poses;
	std::vector<std::vector<float>> model=sampledPC;
	float r = scenesamplingRelative * sceneDiameter;
	float rs = r * r;
	for (int i = 0; i < poses.size(); i++) {
		std::vector<std::vector<float>> curMod;
		curMod = transformPose(model,poses[i].pose);
		float score = 0;
		for (int j = 0; j < curMod.size(); j++) {
			float dist_sum=0;
			for (int count = 0; count < scene.size(); count++) {
				std::vector<float> p1;
				std::vector<float> p2;
				p1.push_back(scene[count][0]);
				p1.push_back(scene[count][1]);
				p1.push_back(scene[count][2]);
				p2.push_back(curMod[j][0]);
				p2.push_back(curMod[j][1]);
				p2.push_back(curMod[j][2]);
				float dist = compute_distance_square(p1, p2);
				if (dist < rs) {
					dist_sum++;
				}
			}
			if (dist_sum > 0) {
				score++;
			}
		}
		outPoses[i].numVotes = score / (float)curMod.size();
	}
	return outPoses;
}

void PPF3DDetector::matchScene(std::vector<Pose3D>  &result, bool saveVotersSwitch, bool recomputeScoreSwich, std::vector<std::vector<float>>scene, float sceneFraction) {
	std::map<float, std::vector<std::vector<float>> >  hashTableLoc = hashTable;   //创建map容器作为哈希表
	int refNumLoc = refPointNum;
	int angleRelativeLoc = angleRelative;
	float distanceStepLoc = distanceStep;
	float angleRadiansLoc = angleRadians;
	float sceneStep = floor(1 / sceneFraction);
	std::vector<float>b_box;
	compute_bounding_box(scene, b_box);
	sceneDiameter = b_box[0];
	scenesamplingRelative = 1 / (sceneDiameter / distanceStepLoc);
	std::vector<std::vector<float>> sampledScene;
	
	/*泊松采样-------------------------------------*/
	samplePCpoisson(scene, scenesamplingRelative, sampledScene);
	//std::ofstream out2("samplescene.txt");
	//for (int i = 0; i < sampledScene.size(); i++) {
	//	//std::cout << std::setw(10) << sampledScene[i][0] << '\t' << i << std::endl;
	//	out2 << sampledScene[i][0] << '\n';
	//}
	//out2.close();

	//for (int i = 0; i < sampledScene.size(); i++) {
	//	std::cout << std::setw(10) << sampledScene[i][0] << '\t' << i << std::endl;
	//}
	/*泊松采样-------------------------------------*/

	float sceneSize = sampledScene.size();
	std::cout << "the sampled SCENE pc size:" << sceneSize << std::endl;

	//为每个场景点分配三个位姿假设
	std::vector<Pose3D>  poseList_temp;

	float posesize = 3 * round(sceneSize / sceneStep);
	for (int pose_NUM = 0; pose_NUM < posesize; pose_NUM++) {
		Pose3D pose_empty;
		poseList_temp.push_back(pose_empty);
	}
	std::cout << "scene: " << sceneSize / sceneStep << "\tposeList: " << posesize << '\n';
	int posesAdded = 0;
	for (int count = 0; count < sceneSize; ) {
		std::cout << "Matching:\t" << count + 1 << "\tof\t" << sceneSize << '\r';
		std::vector<float> p1 = sampledScene[count];
		//初始化计数器，大小为参考点数量*（360 / 12） 论文中n_angle = 30  即360 / 12
		//法线相对于正确法线变化达12°
		//以下关于（refNumLoc+1）的matlab实现里面为什么+1 多分配一份空间，没理解
		float *accumulator = new float[(refNumLoc + 1)*angleRelativeLoc];
		int sizeof_accumulator = (refNumLoc + 1)*angleRelativeLoc;

		for (int i = 0; i < sizeof_accumulator; i++) {
			accumulator[i] = 0;
		}
		//存储位姿结果
		std::vector< std::vector<float>> *coordAccumulator = new std::vector< std::vector<float>>[(refNumLoc + 1)*angleRelativeLoc];

		Eigen::Matrix3f R;
		Eigen::Vector3f t;
		transformRT(p1, R, t);
		for (int ind = 0; ind < sceneSize; ind++) {
			if (count != ind && compute_distance_square(p1, sampledScene[ind]) < pow(modelDiameter, 2)) {
				std::vector<float> p2 = sampledScene[ind];
				std::vector<float> f;
				computePPF(p1, p2, f);
				uint32_t hash = hashPPF(f, angleRadiansLoc, distanceStepLoc);
				Eigen::Vector3f p2_temp(p2[0], p2[1], p2[2]);
				//根据R t变换，相应构成点对的p2也做相同的变换，即整个点对平移
				//这边只对p2进行变换，效果一样，p1用于前面算出R T 矩阵
				Eigen::Vector3f p2t = R * p2_temp + t;
				float alphaScene = atan2(-p2t(2), p2t(1));
				if (sin(alphaScene)*p2t(2) > 0) { alphaScene = -alphaScene; }
				if (hashTableLoc.find(hash) != hashTableLoc.end()) {
					int nNodes = hashTableLoc[hash].size();
					std::vector<std::vector<float>> nodeList;
					for (int i = 0; i < nNodes; i++) {
						nodeList.push_back(hashTableLoc[hash][i]);
					}
					for (int i = 0; i < nNodes; i++) {
						int modelI = nodeList[i][0];
						int ppfInd = nodeList[i][1];
						float alphaModel = nodeList[i][2];
						//模型点对alpha m 也是通过第二点的反三角计算
						//接着两个角度做简单差值，与论文一致，即旋转alphaAngl角度即可实现模型与场景点对的匹配
						float alphaAngle = alphaModel - alphaScene;
						if (alphaAngle > pi) {
							alphaAngle = alphaAngle - 2 * pi;
						}
						else if (alphaAngle < (-pi)) {
							alphaAngle = alphaAngle + 2 * pi;
						}
						//计算alpha的索引，后面进行投票和记录
						int alphaIndex = round((angleRelativeLoc - 1)*(alphaAngle + pi) / (2 * pi));
						int accuIndex = (modelI + 1) * angleRelativeLoc + alphaIndex;
						float voteValue = nodeList[i][3];
						accumulator[accuIndex] = accumulator[accuIndex] + voteValue;
						if (saveVotersSwitch) {
							std::vector<float> coord_A(2);
							coord_A[0] = ind;
							coord_A[1] = ppfInd;
							coordAccumulator[accuIndex].push_back(coord_A);
						}
					}

				}
			}
		}
		/*---------------至此，投票完成---------------*/
		float accuMax = 0;
		int p;
		for (int i = 0; i < sizeof_accumulator; i++) {
			if (accumulator[i] > accuMax) {
				accuMax = accumulator[i];
				p = i;
			}
		}
		//std::cout << p << std::endl;
		accuMax = accuMax * 0.95;
		for (int j = 0; j < sizeof_accumulator; j++) {
			int alphaMaxInd;
			int iMaxInd;
			if (accumulator[j] > accuMax) {
				int peak = j;
				//如果存在大于阈值的peak，则提取模型点的索引值  peak余30，即在(第i对)中的第？个
				alphaMaxInd = peak % angleRelativeLoc;
				//计算出为(第i个)参考点
				iMaxInd = (peak - alphaMaxInd) / angleRelativeLoc - 1;

				Eigen::Matrix3f iR = R.transpose();
				Eigen::Vector3f it = (-1)*iR * t;
				Eigen::Matrix4f iT;
				iT << iR(0, 0), iR(0, 1), iR(0, 2), it(0), iR(1, 0), iR(1, 1), iR(1, 2), it(1), iR(2, 0), iR(2, 1), iR(2, 2), it(2), 0, 0, 0, 1;


				std::vector<float> pMax = sampledPC[iMaxInd];
				Eigen::Matrix3f Rmax;
				Eigen::Vector3f tmax;
				transformRT(pMax, Rmax, tmax);
				//Matrix按行压入，按列存储(0)-(n)。
				Eigen::Matrix4f TMax;
				TMax << Rmax(0, 0), Rmax(0, 1), Rmax(0, 2), tmax(0), Rmax(1, 0), Rmax(1, 1), Rmax(1, 2), tmax(1), Rmax(2, 0), Rmax(2, 1), Rmax(2, 2), tmax(2), 0, 0, 0, 1;
				float alphaAngle = (2 * pi)*alphaMaxInd / (angleRelativeLoc - 1) - pi;

				Eigen::Matrix4f Talpha = XrotMat(alphaAngle);
				Eigen::Matrix4f Tpose = iT * (Talpha *TMax);
				float numVotes = accumulator[peak];
				Pose3D newPose(alphaAngle, posesAdded, numVotes);
				newPose.updatePose(newPose, Tpose);
				/*------存储场景中和模型中的对应点对，未写完*/
				//if (saveVotersSwitch)
				//{
				//	std::vector<std::vector<float>> voted;
				//	for (int vote_coor = 0; vote_coor < coordAccumulator[peak].size(); vote_coor++) {
				//		voted.push_back(coordAccumulator[peak][vote_coor]);
				//	}
				//	for (int voted_num = 0; voted_num < voted.size(); voted_num++) {
				//		float modelI = floor(voted[voted_num][1] / refNumLoc)+1;
				//		float modelY = (int)voted[voted_num][1] % refNumLoc +1;
				//	}
				//}

				poseList_temp[posesAdded] = newPose;
				posesAdded++;
			}
		}
		count += 5;
		delete[] accumulator;
	}
	std::cout << "\nPoses:\t" << posesAdded+1 << '\n';
	for (int i = 0; i < posesize; i++) {
		if (poseList_temp[i].alpha == 0 && poseList_temp[i].modelIndex == 0 && poseList_temp[i].numVotes == 0) {
		}
		else {
			poseList.push_back(poseList_temp[i]);
		}
	}
	std::vector<std::vector<Pose3D>> cluster;
	std::vector<float> votes;
	clusterPoses(poseList, cluster, votes);
	std::cout << "Cluster Poses: " << cluster.size() << '\n';
	std::vector<Pose3D>  resultOrig;
	std::vector<Pose3D>  Cluster_avg;
	averageClusters(cluster, votes, Cluster_avg);
	sortPoses(Cluster_avg, resultOrig);
	if (recomputeScoreSwich) {
		sortPoses(recomputeScore(resultOrig, sampledScene), result);
	}
	else {
		result = resultOrig;
	}
}



void Writefile(std::string filename, std::vector<std::vector<float>> resPC) {
	
	std::ofstream out(filename);
	for (int i = 0; i < resPC.size();i++) {
		out << resPC[i][0] <<'\t'<< resPC[i][1] << '\t' << resPC[i][2] << '\t' << resPC[i][3] << '\t' << resPC[i][4] << '\t' << resPC[i][5] << '\n';
	}
	out.close();
}




#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <boost/thread/thread.hpp>

//---------------读取txt文件-------------------
void CreateCloudFromTxt(const std::string& file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	std::ifstream file(file_path.c_str());//c_str()：生成一个const char*指针，指向以空字符终止的数组。
	std::string line;
	pcl::PointXYZ point;
	while (getline(file, line)) {
		std::stringstream ss(line);
		ss >> point.x;
		ss >> point.y;
		ss >> point.z;
		cloud->push_back(point);
	}
	file.close();
}


void visualization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));

	// 添加需要显示的点云数据

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(cloud1, 255, 255, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud1, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud1, single_color1, "cloud1");
	viewer->addPointCloud<pcl::PointXYZ>(cloud2, single_color2, "cloud2");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");

	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void main(int argc, char* argv[])
{
	std::vector<std::vector<float>> model;
	std::vector<std::vector<float>> scene;
	//std::vector<std::vector<float>> model_sam;
	//std::vector<std::vector<float>> scene_sam;

	time_t start, end;
	start = clock();

	readPLY(model, argv[1]);//命令参数中分别输入模型和场景
	readPLY(scene, argv[2]);
	//readPLY(model_sam, argv[3]);
	//readPLY(scene_sam, argv[4]);

	PPF3DDetector MODEL;
	MODEL.trainModel(model, 0.04);
	std::vector<Pose3D>  result;
	MODEL.matchScene(result,true, false,scene, 0.2);
	std::vector<std::vector<float>> resPC;



	resPC=transformPose(model, result[0].pose);

	std::string filename = "out.txt";
	
	Writefile(filename, resPC);


	
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud3(new pcl::PointCloud<pcl::PointXYZ>);
	CreateCloudFromTxt(argv[1], model_cloud1);
	CreateCloudFromTxt(argv[2], scene_cloud);
	CreateCloudFromTxt("out.txt", model_cloud2);

	std::cout << "running fine registration...\n";
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaxCorrespondenceDistance(100);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(1e-3);
	icp.setMaximumIterations(1000);
	icp.setInputSource(model_cloud2);
	icp.setInputTarget(scene_cloud);
	icp.align(*model_cloud3);
	std::cout << "registration has converged:" << icp.hasConverged() << " score: " <<icp.getFitnessScore() << std::endl;

	end = clock();
	double total_time = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "detection costs: " << total_time << "s\n";

	//visualization(scene_cloud, model_cloud);

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(argc, argv, "Pose estimation demo"));
	int vp_1, vp_2;
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, vp_1);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, vp_2);
	viewer->setBackgroundColor(0, 0, 0, vp_1);
	viewer->setBackgroundColor(0, 0, 0, vp_2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(scene_cloud, 255, 255, 255);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(model_cloud1, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color3(model_cloud3, 255, 0, 0);
	viewer->addPointCloud(scene_cloud, single_color1, "scene1", vp_1);
	viewer->addPointCloud(model_cloud1, single_color2, "model1", vp_1);
	viewer->addPointCloud(scene_cloud, single_color1, "scene2", vp_2);
	viewer->addPointCloud(model_cloud3, single_color3, "model2", vp_2);
	viewer->addText("before", 20, 10, 1, 1, 0, "before", vp_1);
	viewer->addText("after", 20, 10, 1, 1, 0, "after", vp_2);
	viewer->spin();

	system("pause");

}




