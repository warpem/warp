#include "CustomModules.h"

#include <torch/nn/init.h>

torch::nn::Conv2dOptions conv_options(int64_t in_planes, 
                                      int64_t out_planes, 
                                      int64_t kernel_size,
                                      int64_t stride = 1, 
                                      int64_t padding = 0,
                                      bool with_bias = false) 
{
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .bias(with_bias);

    return conv_options;
}

struct BasicBlock : torch::nn::Module 
{
    static const int expansion = 1;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential downsample;

    BasicBlock(int64_t inchannels, 
               int64_t channels, 
               int64_t stride_ = 1,
               torch::nn::Sequential downsample_ = torch::nn::Sequential())
               : conv1(conv_options(inchannels, channels, 3, stride_, 1)),
                 bn1(torch::nn::BatchNorm2dOptions(channels).momentum(0.01)),
                 conv2(conv_options(channels, channels, 3, 1, 1)),
                 bn2(torch::nn::BatchNorm2dOptions(channels).momentum(0.01)),
                 downsample(downsample_)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        stride = stride_;

        if (!downsample->is_empty())
            register_module("downsample", downsample);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->is_empty()) {
            residual = downsample->forward(residual);
        }

        x += residual;
        x = torch::leaky_relu(x, 0.2);

        return x;
    }
};

struct BottleNeck : torch::nn::Module 
{
    static const int expansion = 4;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm2d bn3;
    torch::nn::Sequential downsample;

    BottleNeck(int64_t inchannels, 
               int64_t channels, 
               int64_t stride_ = 1,
               torch::nn::Sequential downsample_ = torch::nn::Sequential())
               
               : conv1(conv_options(inchannels, channels, 1)),
                 bn1(torch::nn::BatchNorm2dOptions(channels).momentum(0.01)),
                 conv2(conv_options(channels, channels, 3, stride_, 1)),
                 bn2(torch::nn::BatchNorm2dOptions(channels).momentum(0.01)),
                 conv3(conv_options(channels, channels * expansion, 1)),
                 bn3(torch::nn::BatchNorm2dOptions(channels * expansion).momentum(0.01)),
                 downsample(downsample_)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        stride = stride_;
        if (!downsample->is_empty())
            register_module("downsample", downsample);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);
        x = torch::relu(x);

        x = conv3->forward(x);
        x = bn3->forward(x);

        if (!downsample->is_empty())
            residual = downsample->forward(residual);

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

template <class Block> struct ResNet : torch::nn::Module 
{
    int64_t inchannels = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc;

    ResNet(int64_t size_input, torch::IntList layers, int64_t num_classes = 2)

           : conv1(conv_options(1, 64, 7, 2, 3)),
             bn1(torch::nn::BatchNorm2dOptions(64).momentum(0.01)),
             layer1(_make_layer(64, layers[0])),
             layer2(_make_layer(128, layers[1], 2)),
             layer3(_make_layer(256, layers[2], 2)),
             layer4(_make_layer(512, layers[3], 2)),
             fc(512 * Block::expansion * pow((size_input / 2 / 2 / 2 / 2) - 6, 2), num_classes)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);

        // Initializing weights
        for (auto m : this->modules(false))
            if (m->name() == "torch::nn::Conv2dImpl")
                for (auto p : m->parameters())
                    torch::nn::init::xavier_normal_(p);
            else if (m->name() == "torch::nn::BatchNorm2dImpl")
                for (auto p : m->named_parameters())
                    if (p.key() == "weight")
                        torch::nn::init::constant_(p.value(), 1);
                    else if (p.key() == "bias")
                        torch::nn::init::constant_(p.value(), 0);
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::leaky_relu(x, 0.2);
        //x = torch::max_pool2d(x, 3, 2, 1);

        x.print();

        x = layer1->forward(x);
        x.print();
        x = layer2->forward(x);
        x.print();
        x = layer3->forward(x);
        x.print();
        x = layer4->forward(x);
        x.print();

        x = torch::avg_pool2d(x, 7, 1);
        x.print();
        x = x.view({ x.sizes()[0], -1 });
        x.print();
        x = fc->forward(x);

        return x;
    }


private:
    torch::nn::Sequential _make_layer(int64_t channels, int64_t blocks, int64_t stride = 1) 
    {
        torch::nn::Sequential downsample;
        if (stride != 1 or inchannels != channels * Block::expansion)
            downsample = torch::nn::Sequential(torch::nn::Conv2d(conv_options(inchannels, channels * Block::expansion, 2, stride)),
                                               torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels * Block::expansion).momentum(0.01)));

        torch::nn::Sequential layers;
        layers->push_back(Block(inchannels, channels, stride, downsample));

        inchannels = channels * Block::expansion;
        for (int64_t i = 0; i < blocks; i++)
            layers->push_back(Block(inchannels, channels));

        return layers;
    }
};



NNModule THSNN_ResNet_ctor(const int64_t size_input, const int64_t blocks1, const int64_t blocks2, const int64_t blocks3, const int64_t blocks4, const int64_t num_classes, NNAnyModule* outAsAnyModule)
{
    //CATCH_RETURN_NNModule
    //(
    ResNet<BasicBlock> Net(size_input, { blocks1, blocks2, blocks3, blocks4 }, num_classes);
    auto mod = std::make_shared<ResNet<BasicBlock>>(Net);

    // Keep a boxed version of the module in case we add it to a Sequential later (the C++ templating means
    // a Module can only be boxed to AnyModule at the point its static type is known).
    if (outAsAnyModule != NULL)
    {
        auto wrapped = std::make_shared<torch::nn::AnyModule>(torch::nn::ModuleHolder<ResNet<BasicBlock>>(*mod));
        *outAsAnyModule = new std::shared_ptr<torch::nn::AnyModule>(wrapped);
    }

    auto res = new std::shared_ptr<torch::nn::Module>(mod);
    return res;
    //);
}

Tensor THSNN_ResNet_forward(const NNModule module, const Tensor tensor)
{
    CATCH_TENSOR((*module)->as<ResNet<BasicBlock>>()->forward(*tensor));
}
