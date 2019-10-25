using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.ML;
using ml.Tutorial;

namespace ml
{
    class Program
    {
        static void Main(string[] args)
        {
            IEnumerable<IMLExample> examples = AllML();

            Output(examples);

            Try(examples);

            Console.WriteLine("Конец");
            Console.ReadLine();
        }

        private static void Try(IEnumerable<IMLExample> examples)
        {
            if (int.TryParse(Console.ReadLine(), out int index))
            {
                var example = examples.ElementAtOrDefault(index);
                example?.Try();
            }
        }

        private static void Output(IEnumerable<IMLExample> examples)
        {
            int i = 0;
            examples.AsParallel().AsOrdered().ForAll(example =>
            {
                Console.WriteLine($"{i++} - {example.GetType().ToString()} {example.Description}");
            });
        }

        private static IEnumerable<IMLExample> AllML()
        {
            return (from t in Assembly.GetExecutingAssembly().GetTypes()
                   where t.GetInterface(nameof(IMLExample)) != null && !t.IsAbstract
                   select Activator.CreateInstance(t)).OfType<IMLExample>().ToList();
        }
    }
}
