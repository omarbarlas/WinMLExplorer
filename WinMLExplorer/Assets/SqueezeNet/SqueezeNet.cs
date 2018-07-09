using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.AI.MachineLearning.Preview;

// squeezenet_old

namespace WinMLExplorer
{
    public sealed class Squeezenet_oldModelInput
    {
        public VideoFrame data_0 { get; set; }
    }

    public sealed class Squeezenet_oldModelOutput
    {
        public IList<float> softmaxout_1 { get; set; }
        public Squeezenet_oldModelOutput()
        {
            this.softmaxout_1 = new List<float>();
        }
    }

    public sealed class Squeezenet_oldModel
    {
        private LearningModelPreview learningModel;
        public static async Task<Squeezenet_oldModel> CreateSqueezenet_oldModel(StorageFile file)
        {
            LearningModelPreview learningModel = await LearningModelPreview.LoadModelFromStorageFileAsync(file);
            Squeezenet_oldModel model = new Squeezenet_oldModel();
            model.learningModel = learningModel;
            return model;
        }
        public async Task<Squeezenet_oldModelOutput> EvaluateAsync(Squeezenet_oldModelInput input) {
            Squeezenet_oldModelOutput output = new Squeezenet_oldModelOutput();
            LearningModelBindingPreview binding = new LearningModelBindingPreview(learningModel);
            binding.Bind("data_0", input.data_0);
            binding.Bind("softmaxout_1", output.softmaxout_1);
            LearningModelEvaluationResultPreview evalResult = await learningModel.EvaluateAsync(binding, string.Empty);
            return output;
        }
    }
}
