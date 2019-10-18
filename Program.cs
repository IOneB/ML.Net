using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using ml.Tutorial;

namespace ml
{
    class Program
    {
        static void Main(string[] args)
        {
            List<IMLExample> examples = new List<IMLExample>
            {
                new Review.ReviewExample(),
                new Tutorial.Binary.BinaryClassification(),
                new Tutorial.MultiClass.MultiClassification()
            };

            Parallel.For(0, examples.Count, (i, state) =>
            {
                Console.WriteLine($"{i} - {examples[i].GetType().ToString()} {examples[i].Description}");
            });

            var example = examples[GetExample(examples.Count)];

            example.Try();
            Console.ReadLine();
        }

        private static int GetExample(int? count)
        {
            if (!count.HasValue)
                return int.Parse(Console.ReadLine());
            return count.Value - 1;
        }
    }
}
