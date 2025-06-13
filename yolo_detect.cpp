// TO COMPILE: y8_detect.cpp  ──  g++ -std=c++17 y8_detect.cpp -o y8 `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "yolo_detect.hpp"
#include <fstream>
#include <iostream>
#include <tuple>


static std::vector<std::string> loadNames(const std::string& file)
{
    std::vector<std::string> v;  std::ifstream ifs(file);
    for(std::string s; std::getline(ifs,s);) if(!s.empty()) v.push_back(s);
    return v;
}

// letter-box to square, returning scale & pad so we can reverse coords
static cv::Mat letterbox(const cv::Mat& img,int newSz,float& gain,int& dx,int& dy)
{
    int h=img.rows, w=img.cols;
    gain = newSz / float(std::max(h,w));
    int nw=int(w*gain), nh=int(h*gain);

    cv::Mat r; cv::resize(img,r,{nw,nh});
    cv::Mat canvas(newSz,newSz,CV_8UC3,cv::Scalar(114,114,114));
    dx=(newSz-nw)/2; dy=(newSz-nh)/2;
    r.copyTo(canvas(cv::Rect(dx,dy,nw,nh)));
    return canvas;
}

// ---------- global singletons (lazy-init) -----------------------------------
static cv::dnn::Net& net()
{
    static cv::dnn::Net n = cv::dnn::readNetFromONNX("yolov8n.onnx");
    n.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    n.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return n;
}
static std::vector<std::string>& names()
{
    static std::vector<std::string> n = loadNames("coco.names");
    return n;
}

// ---------- main API --------------------------------------------------------
std::vector<std::tuple<std::string,int,int>>
detect(const std::string& imgPath)
{
    std::vector<std::tuple<std::string,int,int>> out;

    cv::Mat img=cv::imread(imgPath);
    if(img.empty()){std::cerr<<"bad image: "<<imgPath<<"\n"; return out;}

    float gain; int dx,dy;
    cv::Mat inp = letterbox(img,640,gain,dx,dy);

    cv::Mat blob=cv::dnn::blobFromImage(inp,1/255.0,{640,640},cv::Scalar(),true,false);
    net().setInput(blob);

    std::vector<cv::Mat> outs; net().forward(outs,net().getUnconnectedOutLayersNames());
    cv::Mat o=outs[0];                        // (1,84,8400)  or (1,8400,84)

    // layout fix
    if(o.dims==3 && o.size[0]==1 && o.size[1]<o.size[2]){  // (1,84,8400)
        o = o.reshape(1,o.size[1]); cv::transpose(o,o);    // (8400,84)
    } else {                                               // (1,8400,84)
        o = o.reshape(1,o.size[2]);                        // (8400,84)
    }

    const int rows=o.rows, nc=o.cols-4;        // 4 box + 80 classes
    float* data=(float*)o.data;

    std::vector<int>   ids;  std::vector<float> confs;  std::vector<cv::Rect> boxes;
    const float confTh = 0.40f;     // drop low‑confidence duplicates
    const float nmsTh  = 0.45f;

    for(int i=0;i<rows;++i,data+=o.cols)
    {
        // best class score
        cv::Mat scores(1,nc,CV_32F,data+4);
        cv::Point cid; double best; cv::minMaxLoc(scores,nullptr,&best,nullptr,&cid);
        if(best<confTh) continue;

        float cx=data[0],  cy=data[1],  w=data[2],  h=data[3];   // already in 640‑px space

        // reverse to original image
        float cx0=(cx-dx)/gain, cy0=(cy-dy)/gain, w0=w/gain, h0=h/gain;
        int   left=int(cx0-w0/2), top=int(cy0-h0/2);

        ids.push_back(cid.x); confs.push_back((float)best);
        boxes.emplace_back(left,top,int(w0),int(h0));
    }

    std::vector<int> keep; cv::dnn::NMSBoxes(boxes,confs,confTh,nmsTh,keep);
    for(int k:keep){
        const auto& b=boxes[k];
        int cx=b.x+b.width/2, cy=b.y+b.height/2;
        out.emplace_back(names()[ids[k]],cx,cy);
    }
    return out;
}

// --------------------------- tiny demo --------------------------------------
#ifdef DEMO
int main(int argc,char**argv)
{
    if(argc<2){std::cerr<<"usage: "<<argv[0]<<" image.jpg\n"; return 1;}
    for(auto& t:detect(argv[1]))
        std::cout<<std::get<0>(t)<<' '<<std::get<1>(t)<<' '<<std::get<2>(t)<<"\n";
}
#endif