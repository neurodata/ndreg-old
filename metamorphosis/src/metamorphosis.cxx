#include "itkDebug.h"
#include <iomanip>   // setprecision()
#include <algorithm> // min(), max()
#include "itkNumericTraits.h"
#include "itkCommand.h"
#include "itkTimeProbe.h"
#include "itkImage.h"
#include "itkCommandLineArgumentParser.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkBSplineKernelFunction.h"
#include "itkGridImageSource.h"
#include "itkMetamorphosisImageRegistrationMethodv4.h"

using namespace std;

typedef itk::CommandLineArgumentParser  ParserType;
template<typename TMetamorphosis> class MetamorphosisObserver;
template<typename TImage> int Metamorphosis(typename ParserType::Pointer parser);

int main(int argc, char* argv[])
{
  /** Check command line arguments */
  ParserType::Pointer parser = ParserType::New();
  parser->SetCommandLineArguments(argc,argv);

  if( !(parser->ArgumentExists("--input") && parser->ArgumentExists("--reference") && parser->ArgumentExists("--output")) )
  {
    cerr<<"Usage:"<<endl;
    cerr<<"\t"<<argv[0]<<" --input InputPath --reference ReferencePath --output OutputPath"<<endl;
    cerr<<"\t\t[ --weight InputWeightPath"<<endl;
    cerr<<"\t\t  --displacement OutputDisplacementPath"<<endl;
    cerr<<"\t\t  --bias OutputBiasPath"<<endl;
    cerr<<"\t\t  --grid OutputGridPath"<<endl;
    cerr<<"\t\t  --alpha RegistrationSmoothness"<<endl;
    cerr<<"\t\t  --beta BiasSmoothness"<<endl;
    cerr<<"\t\t  --sigma Sigma"<<endl;
    cerr<<"\t\t  --mu Mu"<<endl;
    cerr<<"\t\t  --steps NumberOfTimesteps"<<endl;
    cerr<<"\t\t  --iterations MaximumNumberOfIterations"<<endl;
    cerr<<"\t\t  --verbose ]"<<endl;
    cerr<<"Note: To disable bias correction use a BiasSmoothness < 0."<<endl;
    return EXIT_FAILURE;
  }

  /** Read input (moving) image information*/
  string inputPath;
  parser->GetCommandLineArgument("--input",inputPath);
  /*
  itk::ImageIOBase::Pointer io = itk::ImageIOFactory::CreateImageIO(inputPath.c_str(), itk::ImageIOFactory::ReadMode);
  try
  {
    io->ReadImageInformation();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Could not read input image: "<<inputPath<<endl;
    cerr<<exceptionObject<<endl;
    return EXIT_FAILURE;
  }
  */
  /** Run Metamorphosis */
  typedef float      PixelType;
  switch(3)//io->GetNumberOfDimensions())
  {
    case 2:
      // Run 2D Metamorphosis
      typedef itk::Image<PixelType,2> Image2DType;
      return Metamorphosis<Image2DType>(parser);
    case 3:
      // Run 3D Metamorphosis
      typedef itk::Image<PixelType,3> Image3DType;
      return Metamorphosis<Image3DType>(parser);
    default:
      cerr<<"Error only 2D and 3D images are supported.";
      return EXIT_FAILURE;
  }

} // end main

template<typename TImage>
int Metamorphosis(typename ParserType::Pointer parser)
{
  /** Construct Metamorphosis */
  typedef TImage ImageType;
  typedef itk::MetamorphosisImageRegistrationMethodv4<ImageType,ImageType,ImageType>  MetamorphosisType;
  typename MetamorphosisType::Pointer metamorphosis = MetamorphosisType::New();

  /** Set input (moving) image */
  string inputPath;
  parser->GetCommandLineArgument("--input",inputPath);

  typedef itk::ImageFileReader<ImageType> ReaderType;
  typename ReaderType::Pointer inputReader = ReaderType::New();
  inputReader->SetFileName(inputPath);
  try
  {
    inputReader->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Could not read input image: "<<inputPath<<endl;
    cerr<<exceptionObject<<endl;
    return EXIT_FAILURE;
  }

  typename ImageType::Pointer movingImage = inputReader->GetOutput();
  metamorphosis->SetMovingImage(movingImage); // I_0

  /** Set reference (fixed) image */
  string referencePath;
  parser->GetCommandLineArgument("--reference",referencePath);

  typename ReaderType::Pointer referenceReader = ReaderType::New();
  referenceReader->SetFileName(referencePath);
  try
  {
    referenceReader->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Could not read reference image: "<<referencePath<<endl;
    cerr<<exceptionObject<<endl;
    return EXIT_FAILURE;
  }

  typename ImageType::Pointer fixedImage = referenceReader->GetOutput();
  metamorphosis->SetFixedImage(fixedImage);   // I_1

  if(parser->ArgumentExists("--weight"))
  {
    /** Set weight image */
    string weightPath;
    parser->GetCommandLineArgument("--weight",weightPath);
    
    typename ReaderType::Pointer weightReader = ReaderType::New();
    weightReader->SetFileName(weightPath);
    try
    {
      weightReader->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not weight image: "<<weightPath<<endl;
      cerr<<exceptionObject<<endl;
      return EXIT_FAILURE;
    }

    metamorphosis->SetWeightImage(weightReader->GetOutput());
  }

  /** Set metamorphosis parameters */
  if(parser->ArgumentExists("--alpha"))
  {
    double alpha;
    parser->GetCommandLineArgument("--alpha",alpha);
    metamorphosis->SetRegistrationSmoothness(alpha);
  }

  if(parser->ArgumentExists("--beta"))
  {
    double beta;
    parser->GetCommandLineArgument("--beta",beta);

    if(beta < 0)
    {
      metamorphosis->BiasCorrectionOff();
    }
    else
    {
      metamorphosis->SetBiasSmoothness(beta);
    }
  }

  if(parser->ArgumentExists("--sigma"))
  {
    double sigma;
    parser->GetCommandLineArgument("--sigma",sigma);
    metamorphosis->SetSigma(sigma);
  }

  if(parser->ArgumentExists("--mu"))
  {
    double mu;
    parser->GetCommandLineArgument("--mu",mu);
    metamorphosis->SetMu(mu);
  }

  if(parser->ArgumentExists("--steps"))
  {
    unsigned int numberOfTimeSteps;
    parser->GetCommandLineArgument("--steps",numberOfTimeSteps);
    metamorphosis->SetNumberOfTimeSteps(numberOfTimeSteps);
  }

  if(parser->ArgumentExists("--iterations"))
  {
    unsigned int numberOfIterations;
    parser->GetCommandLineArgument("--iterations",numberOfIterations);
    metamorphosis->SetNumberOfIterations(numberOfIterations);
  }

  if(parser->ArgumentExists("--verbose"))
  {
    typedef MetamorphosisObserver<MetamorphosisType>  MetamorphosisObserverType;
    typename MetamorphosisObserverType::Pointer observer = MetamorphosisObserverType::New();
    metamorphosis->AddObserver(itk::IterationEvent(),observer);
  }

  /** Run metamorphosis */
  itk::TimeProbe clock;
  clock.Start();
  try
  {
    metamorphosis->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Metamorphsis did not terminate normally."<<endl;
    cerr<<exceptionObject<<endl;
    return EXIT_FAILURE;
  }
  clock.Stop();

  cout<<"E = "<<metamorphosis->GetEnergy()<<" ("<<metamorphosis->GetEnergy()/metamorphosis->GetInitialEnergy()*100<<"%)"<<endl;
  cout<<"Time = "<<clock.GetTotal()<<"s"<<" ("<<clock.GetTotal()/60<<"m)"<<endl;

  /** Write output images */
  int returnValue = EXIT_SUCCESS;

  // Compute I_0 o \phi_{10}
  typedef typename MetamorphosisType::OutputTransformType TransformType;
  typename TransformType::Pointer transform = const_cast<TransformType*>(metamorphosis->GetOutput()->Get()); // \phi_{10}

  typedef itk::ResampleImageFilter<ImageType,ImageType,typename TransformType::ScalarType>   OutputResamplerType;
  typename OutputResamplerType::Pointer outputResampler = OutputResamplerType::New();
  outputResampler->SetInput(movingImage);   // I_0
  outputResampler->SetTransform(transform); // \phi_{10}
  outputResampler->UseReferenceImageOn();
  outputResampler->SetReferenceImage(fixedImage);

  // Compute I(1) = I_0 o \phi_{10} + B(1)
  typedef itk::AddImageFilter<ImageType,ImageType,ImageType>  AdderType;
  typename AdderType::Pointer adder = AdderType::New();
  adder->SetInput1(outputResampler->GetOutput());  // I_0 o \phi_{10}
  adder->SetInput2(metamorphosis->GetBias());      // B(1)
  adder->Update();

  // Limit intensity of I(1) to intensity range of ouput image type
  itkStaticConstMacro(ImageDimension, unsigned int, ImageType::ImageDimension);
  typedef unsigned char   OutputPixelType;
  typedef itk::Image<OutputPixelType,ImageDimension>  OutputImageType;

  typedef itk::MinimumMaximumImageCalculator<ImageType>  CalculatorType;
  typename CalculatorType::Pointer calculator = CalculatorType::New();
  calculator->SetImage(adder->GetOutput());  // I(1)
  calculator->Compute();

  double outputMinimum = std::max((typename ImageType::PixelType)itk::NumericTraits<OutputPixelType>::min(),calculator->GetMinimum());
  double outputMaximum = std::min((typename ImageType::PixelType)itk::NumericTraits<OutputPixelType>::max(),calculator->GetMaximum());

  typedef itk::IntensityWindowingImageFilter<ImageType,OutputImageType>  IntensityWindowerType;
  typename IntensityWindowerType::Pointer intensityWindower = IntensityWindowerType::New();
  intensityWindower->SetInput(adder->GetOutput()); // I(1)
  intensityWindower->SetWindowMinimum(outputMinimum);
  intensityWindower->SetWindowMaximum(outputMaximum);
  intensityWindower->SetOutputMinimum(outputMinimum);
  intensityWindower->SetOutputMaximum(outputMaximum);

  // Write output image, I(1)
  string outputPath;
  parser->GetCommandLineArgument("--output",outputPath);

  typedef itk::ImageFileWriter<OutputImageType>  OutputWriterType;
  typename OutputWriterType::Pointer outputWriter = OutputWriterType::New();
  outputWriter->SetInput(intensityWindower->GetOutput()); // I(1)
  outputWriter->SetFileName(outputPath);
  try
  {
    outputWriter->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    cerr<<"Error: Could not write output image: "<<outputPath<<endl;
    cerr<<exceptionObject<<endl;
    returnValue = EXIT_FAILURE;
  }

  // Write displacement, \phi_{10}
  string displacementPath;
  parser->GetCommandLineArgument("--displacement",displacementPath);

  if(displacementPath != "")
  {
    typedef typename TransformType::DisplacementFieldType  DisplacementFieldType;
    typedef itk::ImageFileWriter<DisplacementFieldType>    DisplacementWriterType;
    typename DisplacementWriterType::Pointer displacementWriter = DisplacementWriterType::New();
    displacementWriter->SetInput(transform->GetDisplacementField()); // \phi_{10}
    displacementWriter->SetFileName(displacementPath);
    try
    {
      displacementWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write displacement field: "<<displacementPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write bias, B(1)
  string biasPath;
  parser->GetCommandLineArgument("--bias",biasPath);

  if(biasPath != "")
  {
    typedef float                                       FloatPixelType;
    typedef itk::Image<FloatPixelType,ImageDimension>   FloatImageType;

    typedef itk::CastImageFilter<typename MetamorphosisType::ImageType,FloatImageType>  CasterType;
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput(metamorphosis->GetBias()); // B(1)

    typedef itk::ImageFileWriter<FloatImageType>  BiasWriterType;
    typename BiasWriterType::Pointer biasWriter = BiasWriterType::New();
    biasWriter->SetInput(caster->GetOutput());
    biasWriter->SetFileName(biasPath);
    try
    {
      biasWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write bias image: "<<biasPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write grid
  string gridPath;
  parser->GetCommandLineArgument("--grid",gridPath);

  if(gridPath != "")
  {
    typedef itk::BSplineKernelFunction<0>  KernelType;
    typename KernelType::Pointer kernelFunction = KernelType::New();

    typedef itk::GridImageSource<ImageType> GridSourceType;
    typename GridSourceType::Pointer gridSource = GridSourceType::New();
    typename GridSourceType::ArrayType gridSpacing = fixedImage->GetSpacing()*5;
    typename GridSourceType::ArrayType gridOffset; gridOffset.Fill(0.0);
    typename GridSourceType::ArrayType sigma = fixedImage->GetSpacing();
    typename GridSourceType::ArrayType which; which.Fill(true);

    gridSource->SetKernelFunction(kernelFunction);
    gridSource->SetSpacing(movingImage->GetSpacing());
    gridSource->SetOrigin(movingImage->GetOrigin());
    gridSource->SetSize(movingImage->GetLargestPossibleRegion().GetSize());
    gridSource->SetGridSpacing(gridSpacing);
    gridSource->SetGridOffset(gridOffset);
    gridSource->SetWhichDimensions(which);
    gridSource->SetSigma(sigma);
    gridSource->SetScale(itk::NumericTraits<unsigned char>::max());

    typedef itk::ResampleImageFilter<ImageType,OutputImageType,typename TransformType::ScalarType>   GridResamplerType;
    typename GridResamplerType::Pointer gridResampler = GridResamplerType::New();
    gridResampler->SetInput(gridSource->GetOutput());
    gridResampler->SetTransform(transform);
    gridResampler->UseReferenceImageOn();
    gridResampler->SetReferenceImage(fixedImage);

    typename OutputWriterType::Pointer gridWriter = OutputWriterType::New();
    gridWriter->SetInput(gridResampler->GetOutput());
    gridWriter->SetFileName(gridPath);
    try
    {
      gridWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write grid image: "<<gridPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  return returnValue;
} // end Metamorphosis

template<typename TMetamorphosis>
class MetamorphosisObserver: public itk::Command
{
public:
  /** Standard class typedefs. */
  typedef MetamorphosisObserver          Self;
  typedef itk::Command                   Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation throught object factory */
  itkNewMacro(Self);

  /** Filter typedefs */
  typedef TMetamorphosis*             FilterPointer;
  typedef const TMetamorphosis*       FilterConstPointer;
 
  /** Execute for non-const caller */
  void Execute(itk::Object *caller, const itk::EventObject &event)
  {
    FilterPointer filter = dynamic_cast<FilterPointer>(caller);
    ostringstream ss;

    if(itk::IterationEvent().CheckEvent(&event) )
    {
      if(filter->GetCurrentIteration() % 20 == 0) // Print column headings every 20 iterations
      {
        ss<<"\tE, E_velocity, E_rate, E_image, LearningRate"<<std::endl;
      }
      ss<<std::setprecision(3) << std::fixed;
      ss<<filter->GetCurrentIteration()<<".\t"<<filter->GetEnergy()<<", "<<filter->GetVelocityEnergy()<<", "<<filter->GetRateEnergy()<<", "<<filter->GetImageEnergy()<<", ";
      ss.setf(std::ios::scientific,std::ios::floatfield);
      ss<<filter->GetLearningRate()<<std::endl;
      std::cout<<ss.str();
    }    
  }

  /** Execute for non-const caller */
  void Execute(const itk::Object* caller, const itk::EventObject &event)
  {
    FilterConstPointer filter = dynamic_cast<FilterConstPointer>(caller);
  }

protected:
  MetamorphosisObserver(){}; // Constructor
};  // end class MetamorphosisObserver
