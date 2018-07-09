using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.AI.MachineLearning.Preview;
using WinMLExplorer.MLModels;
using WinMLExplorer.Models;
using Windows.UI;
using WinMLExplorer.ViewModels;
using System.IO;
using System.Linq;

// squeezenet_old

namespace WinMLExplorer
{
    public sealed class Squeezenet_oldModelInput
    {
        public VideoFrame data { get; set; }
    }

    public sealed class Squeezenet_oldModelOutput
    {
        public IList<float> softmaxout_1 { get; set; }
        public Squeezenet_oldModelOutput()
        {
            this.softmaxout_1 = new List<float>();
        }
    }

    public sealed class Squeezenet_oldModel : WinMLModel
    {
        public Squeezenet_oldModel()
        {
            LoadLabelsfromJSON();
        }

        private async Task LoadLabelsfromJSON()
        {
            ListOfObjectLabels?.Clear();
            // Parse labels from label file
            var file = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/SqueezeNet/Labels.json"));
            using (var inputStream = await file.OpenReadAsync())
            using (var classicStream = inputStream.AsStreamForRead())
            using (var streamReader = new StreamReader(classicStream))
            {
                string line = "";
                char[] charToTrim = { '\"', ' ' };
                while (streamReader.Peek() >= 0)
                {
                    line = streamReader.ReadLine();
                    line.Trim(charToTrim);
                    var indexAndLabel = line.Split(':');
                    if (indexAndLabel.Count() == 2)
                    {
                        ListOfObjectLabels.Add(indexAndLabel[1]);
                    }
                }
            }
        }

        List<string> ListOfObjectLabels = new List<string>();

        public override string DisplayInputName => "SqueezNet Model";

        public override float DisplayMinProbability => 0.1f;

        public override string DisplayName => "SqueezNet Model Name";

       

        public override DisplayResultSetting[] DisplayResultSettings => new DisplayResultSetting[]
         {
            new DisplayResultSetting() { Name = "Normal", Color = ColorHelper.FromArgb(255, 33, 206, 114), ProbabilityRange = new Tuple<float, float>(0f, 1f) },
            new DisplayResultSetting() { Name = "Defective", Color = ColorHelper.FromArgb(255, 206, 44, 33), ProbabilityRange = new Tuple<float, float>(0f, 1f) }
         };

        public override string Filename => "SqueezeNet.onnx";

        public override string Foldername => "SqueezeNet";

       
        private LearningModelPreview learningModel;
        public static async Task<Squeezenet_oldModel> CreateSqueezenet_oldModel(StorageFile file)
        {
            LearningModelPreview learningModel = await LearningModelPreview.LoadModelFromStorageFileAsync(file);
            Squeezenet_oldModel model = new Squeezenet_oldModel();
            model.learningModel = learningModel;
            return model;
        }

        protected override async Task EvaluateAsync(MLModelResult result, VideoFrame inputFrame)
        {

            



            // Initialize the input
            Squeezenet_oldModelInput input = new Squeezenet_oldModelInput() { data = inputFrame };

            // Evaluate the input
            Squeezenet_oldModelOutput output = await EvaluateAsync(input, result.CorrelationId);

            // Get first label from output
            List<float> resultProbabilities = output.softmaxout_1 as List<float>;

            // Find the result of the evaluation in the bound output (the top classes detected with the max confidence)
            List<float> topProbabilities = new List<float>() { 0.0f, 0.0f, 0.0f };
            List<int> topProbabilityLabelIndexes = new List<int>() { 0, 0, 0 };
            for (int i = 0; i < resultProbabilities.Count; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (resultProbabilities[i] > topProbabilities[j])
                    {
                        topProbabilityLabelIndexes[j] = i;
                        topProbabilities[j] = resultProbabilities[i];
                        break;
                    }
                }
            }



         

            float probability = topProbabilities[0];

            string label = ListOfObjectLabels[topProbabilityLabelIndexes[0]];

            result.OutputFeatures = new MLModelOutputFeature[]
            {
                    new MLModelOutputFeature() { Label = label, Probability = probability }
            };
        }
        public async Task<Squeezenet_oldModelOutput> EvaluateAsync(Squeezenet_oldModelInput input, string correlationId = "")
        {
            Squeezenet_oldModelOutput output = new Squeezenet_oldModelOutput();
            LearningModelBindingPreview binding = new LearningModelBindingPreview(this.LearningModel);
            binding.Bind("data_0", input.data);
            binding.Bind("softmaxout_1", output.softmaxout_1);
            LearningModelEvaluationResultPreview evalResult = await this.LearningModel.EvaluateAsync(binding, correlationId);
            return output;
        }
    }
}
