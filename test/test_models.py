from collections import OrderedDict
from itertools import product
import torch
from torchvision import models
import unittest
import math



def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# model_name, expected to script without error
torchub_models = {
    "deeplabv3_resnet101": False,
    "mobilenet_v2": True,
    "resnext50_32x4d": False,
    "fcn_resnet101": False,
    "googlenet": False,
    "densenet121": False,
    "resnet18": True,
    "alexnet": True,
    "shufflenet_v2_x1_0": True,
    "squeezenet1_0": True,
    "vgg11": True,
    "inception_v3": False,
}

STANDARD_SEED = 1729 # https://fburl.com/3i5wkg9p
STANDARD_INPUT_SHAPE = (1, 3, 224, 224) # for ImageNet-trained models
EPSILON = 1e-6

class Tester(unittest.TestCase):
    
    # set random seed for whatever callable follows (if any)
    # can be called with no args to just set to standard random seed
    def _rand_sync(self, callable=None, **kwargs):
        torch.random.manual_seed(STANDARD_SEED)
        if callable is not None:
            return callable(**kwargs)
    
    # create a randomly-weighted model w/ synced RNG state
    def _get_test_model(self, callable, **kwargs):
        if callable is None:
            assert(False)
        return self._rand_sync(callable, **kwargs).eval()
    
    # create random tensor with given shape using synced RNG state
    def _get_test_input(self, shape):
        return self._rand_sync(lambda: torch.rand(shape))
        
    def check_script(self, model, name):
        if name not in torchub_models:
            return
        scriptable = True
        try:
            torch.jit.script(model)
        except Exception:
            scriptable = False
        self.assertEqual(torchub_models[name], scriptable)
        
    def _check_scriptable(self, model, should_be_scriptable):
        scriptable = True
        try:
            torch.jit.script(model)
        except Exception:
            scriptable = False
        self.assertEqual(should_be_scriptable, scriptable)

    ##
    # Helpers that do some standard stuff, depending on the nature of the model
    #
    
    # so deprecated
    def _test_classification_model(self, name, input_shape):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.__dict__[name](num_classes=50)
        self.check_script(model, name)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)
        
    def _check_classification_output_shape(self, model, test_input, num_classes):
        out = model(test_input)
        self.assertEqual(out.shape, (1, num_classes))

    def _test_segmentation_model(self, name):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.segmentation.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(tuple(out["out"].shape), (1, 50, 300, 300))

    def _test_detection_model(self, name):
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        self.check_script(model, name)
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])

    def _test_video_model(self, name):
        # the default input shape is
        # bs * num_channels * clip_len * h *w
        input_shape = (1, 3, 4, 112, 112)
        # test both basicblock and Bottleneck
        model = models.video.__dict__[name](num_classes=50)
        self.check_script(model, name)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(out.shape[-1], 50)

    def _make_sliced_model(self, model, stop_layer):
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            if name == stop_layer:
                break
        new_model = torch.nn.Sequential(layers)
        return new_model
    
    def _check_model_correctness(self, model, expected_values):
        x = self._rand_sync(lambda: torch.rand(STANDARD_INPUT_SHAPE))
        y = self._rand_sync(lambda: model(x)) # because dropout &c
        for k in expected_values:
            self.assertTrue(abs(y[0][k].item() - expected_values[k]) < EPSILON)
        
    def _scratch_random_check(self, model, x):
        y = self._rand_sync(lambda: model(x))
        print(y.shape)
        indices = [65, 172, 195, 241, 319, 333, 538, 546, 763, 885]
        vals = []
        for i in indices:
            vals.append(math.trunc(1e6 * y[0][i].item()) / 1e6)
        print('{')
        for i in range(len(indices)):
            print('{} : {},'.format(indices[i], vals[i]))
        print('}')
        
    
    ##
    # Classification model tests
    #

    def test_classification_alexnet(self):
        model = self._get_test_model(models.alexnet) # num_classes=1000
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        
        self._check_scriptable(model, True)
        self._check_classification_output_shape(model, test_input, 1000)
        expected_values = { # known good values for this model with rand seeded to standard
            130 : 0.019345,
            257 : -0.002852,
            313 : 0.019647,
            361 : -0.006478,
            466 : 0.011666,
            525 : 0.009539,
            537 : 0.01841,
            606 : 0.003135,
            667 : 0.004638,
            945 : -0.014482
        }
        self._check_model_correctness(model, expected_values)
        
    def test_classification_resnet18(self):
        model = self._get_test_model(models.resnet18) # num_classes=1000
        test_input = self._get_test_input(STANDARD_INPUT_SHAPE)
        
        self._check_scriptable(model, True)
        self._check_classification_output_shape(model, test_input, 1000)
        self._scratch_random_check(model, test_input)
        expected_values = { # known good values for this model with rand seeded to standard
            65 : -0.115954,
            172 : 0.139294,
            195 : 1.248264,
            241 : -1.769466,
            319 : -0.237925,
            333 : -0.038517,
            538 : -0.346574,
            546 : 0.364637,
            763 : 0.43461,
            885 : -1.386981
        }
        self._check_model_correctness(model, expected_values)

class FuckOff:
    def test_classification_resnet34(self):
        self._test_classification_model('resnet34', STANDARD_INPUT_SHAPE)

    def test_classification_resnet50(self):
        self._test_classification_model('resnet50', STANDARD_INPUT_SHAPE)

    def test_classification_resnet101(self):
        self._test_classification_model('resnet101', STANDARD_INPUT_SHAPE)

    def test_classification_resnet152(self):
        self._test_classification_model('resnet152', STANDARD_INPUT_SHAPE)

    def test_classification_resnext50_32x4d(self):
        self._test_classification_model('resnext50_32x4d', STANDARD_INPUT_SHAPE)

    def test_classification_resnext101_32x8d(self):
        self._test_classification_model('resnext101_32x8d', STANDARD_INPUT_SHAPE)

    def test_classification_wide_resnet50_2(self):
        self._test_classification_model('wide_resnet50_2', STANDARD_INPUT_SHAPE)

    def test_classification_wide_resnet101_2(self):
        self._test_classification_model('wide_resnet101_2', STANDARD_INPUT_SHAPE)

    def test_classification_vgg11(self):
        self._test_classification_model('vgg11', STANDARD_INPUT_SHAPE)

    def test_classification_vgg11_bn(self):
        self._test_classification_model('vgg11_bn', STANDARD_INPUT_SHAPE)

    def test_classification_vgg13(self):
        self._test_classification_model('vgg13', STANDARD_INPUT_SHAPE)

    def test_classification_vgg13_bn(self):
        self._test_classification_model('vgg13_bn', STANDARD_INPUT_SHAPE)

    def test_classification_vgg16(self):
        self._test_classification_model('vgg16', STANDARD_INPUT_SHAPE)

    def test_classification_vgg16_bn(self):
        self._test_classification_model('vgg16_bn', STANDARD_INPUT_SHAPE)

    def test_classification_vgg19_bn(self):
        self._test_classification_model('vgg19_bn', STANDARD_INPUT_SHAPE)

    def test_classification_vgg19(self):
        self._test_classification_model('vgg19', STANDARD_INPUT_SHAPE)

    def test_classification_squeezenet1_0(self):
        self._test_classification_model('squeezenet1_0', STANDARD_INPUT_SHAPE)

    def test_classification_squeezenet1_1(self):
        self._test_classification_model('squeezenet1_1', STANDARD_INPUT_SHAPE)

    def test_classification_inception_v3(self):
        self._test_classification_model('inception_v3', (1, 3, 299, 299))

    def test_classification_densenet121(self):
        self._test_classification_model('densenet121', STANDARD_INPUT_SHAPE)

    def test_classification_densenet169(self):
        self._test_classification_model('densenet169', STANDARD_INPUT_SHAPE)

    def test_classification_densenet201(self):
        self._test_classification_model('densenet201', STANDARD_INPUT_SHAPE)

    def test_classification_densenet161(self):
        self._test_classification_model('densenet161', STANDARD_INPUT_SHAPE)

    def test_classification_googlenet(self):
        self._test_classification_model('googlenet', STANDARD_INPUT_SHAPE)

    def test_classification_mobilenet_v2(self):
        self._test_classification_model('mobilenet_v2', STANDARD_INPUT_SHAPE)

    def test_classification_mnasnet0_5(self):
        self._test_classification_model('mnasnet0_5', STANDARD_INPUT_SHAPE)

    def test_classification_mnasnet0_75(self):
        self._test_classification_model('mnasnet0_75', STANDARD_INPUT_SHAPE)

    def test_classification_mnasnet1_0(self):
        self._test_classification_model('mnasnet1_0', STANDARD_INPUT_SHAPE)

    def test_classification_mnasnet1_3(self):
        self._test_classification_model('mnasnet1_3', STANDARD_INPUT_SHAPE)

    def test_classification_shufflenet_v2_x0_5(self):
        self._test_classification_model('shufflenet_v2_x0_5', STANDARD_INPUT_SHAPE)

    def test_classification_shufflenet_v2_x1_0(self):
        self._test_classification_model('shufflenet_v2_x1_0', STANDARD_INPUT_SHAPE)

    def test_classification_shufflenet_v2_x1_5(self):
        self._test_classification_model('shufflenet_v2_x1_5', STANDARD_INPUT_SHAPE)

    def test_classification_shufflenet_v2_x2_0(self):
        self._test_classification_model('shufflenet_v2_x2_0', STANDARD_INPUT_SHAPE)


    def test_memory_efficient_densenet(self):
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)

        for name in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
            model1 = models.__dict__[name](num_classes=50, memory_efficient=True)
            params = model1.state_dict()
            model1.eval()
            out1 = model1(x)
            out1.sum().backward()

            model2 = models.__dict__[name](num_classes=50, memory_efficient=False)
            model2.load_state_dict(params)
            model2.eval()
            out2 = model2(x)

            max_diff = (out1 - out2).abs().max()

            self.assertTrue(max_diff < 1e-5)

    def test_resnet_dilation(self):
        # TODO improve tests to also check that each layer has the right dimensionality
        for i in product([False, True], [False, True], [False, True]):
            model = models.__dict__["resnet50"](replace_stride_with_dilation=i)
            model = self._make_sliced_model(model, stop_layer="layer4")
            model.eval()
            x = torch.rand(1, 3, 224, 224)
            out = model(x)
            f = 2 ** sum(i)
            self.assertEqual(out.shape, (1, 2048, 7 * f, 7 * f))

    def test_mobilenetv2_residual_setting(self):
        model = models.__dict__["mobilenet_v2"](inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape[-1], 1000)


for model_name in []: # get_available_segmentation_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_segmentation_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name in []: # get_available_detection_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_detection_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)


for model_name in []: # get_available_video_models():

    def do_test(self, model_name=model_name):
        self._test_video_model(model_name)

    setattr(Tester, "test_" + model_name, do_test)

if __name__ == '__main__':
    unittest.main()
#    for model_name in get_available_classification_models():
#        print('    def test_classification_{}(self):'.format(model_name))
#        print('        self._test_classification_model({}, STANDARD_INPUT_SHAPE)'.format(model_name))
#        print('')
