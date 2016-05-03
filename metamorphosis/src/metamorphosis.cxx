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
#include "itkClampImageFilter.h"
#include "itkBSplineKernelFunction.h"
#include "itkGridImageSource.h"
#include "itkMetamorphosisImageRegistrationMethodv4.h"

using namespace std;

typedef itk::CommandLineArgumentParser  ParserType;
template<typename TMetamorphosis> class MetamorphosisObserver;
template<typename TImage> int Metamorphosis(typename TImage::Pointer fixedImage, typename ParserType::Pointer parser);

int main(int argc, char* argv[])
{
  /** Check command line arguments */
  ParserType::Pointer parser = ParserType::New();
  parser->SetCommandLineArguments(argc,argv);

  if( !(parser->ArgumentExists("--in") && parser->ArgumentExists("--ref") && parser->ArgumentExists("--out")) )
  {
    cerr<<"Usage:"<<endl;
    cerr<<"\t"<<argv[0]<<" --in InputPath --ref ReferencePath --out OutputPath"<<endl;
    cerr<<"\t\t[ --field OutputDisplacementFieldPath"<<endl;
    cerr<<"\t\t  --invfield OutputInverseDisplacementFieldPath"<<endl;
    cerr<<"\t\t  --bias OutputBiasPath"<<endl;
    cerr<<"\t\t  --grid OutputGridPath"<<endl;
    cerr<<"\t\t  --scale Scale"<<endl;
    cerr<<"\t\t  --alpha RegistrationSmoothness"<<endl;
    cerr<<"\t\t  --beta BiasSmoothness"<<endl;
    cerr<<"\t\t  --sigma Sigma"<<endl;
    cerr<<"\t\t  --mu Mu"<<endl;
    cerr<<"\t\t  --epsilon LearningRate"<<endl;
    cerr<<"\t\t  --fraction MinimumInitialEnergyFraction"<<endl;
    cerr<<"\t\t  --steps NumberOfTimesteps"<<endl;
    cerr<<"\t\t  --iterations MaximumNumberOfIterations"<<endl;
    cerr<<"\t\t  --verbose ]"<<endl;
    cerr<<"Note: To disable bias correction use a BiasSmoothness < 0."<<endl;
    return EXIT_FAILURE;
  }

  /** Read reference image as 3D volume */
  const unsigned int numberOfDimensions = 3;
  typedef float      PixelType;
  typedef itk::Image<PixelType,2>                    Image2DType;
  typedef itk::Image<PixelType,3>                    Image3DType;

  string referencePath;
  parser->GetCommandLineArgument("--ref",referencePath);
  typedef itk::ImageFileReader<Image3DType> ReaderType;
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

  typename Image3DType::Pointer referenceImage = referenceReader->GetOutput();

  /** Setup metamorphosis */
  if(referenceImage->GetLargestPossibleRegion().GetSize()[numberOfDimensions-1] <= 1) // If reference image is 2D...
  {
    // Do 2D metamorphosis
    Image3DType::RegionType region = referenceImage->GetLargestPossibleRegion();
    Image3DType::SizeType   size = region.GetSize();
    size[numberOfDimensions-1] = 0;
    region.SetSize(size);

    typedef itk::ExtractImageFilter<Image3DType, Image2DType> ExtractorType;
    ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(referenceImage);
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();
    extractor->Update();

    return Metamorphosis<Image2DType>(extractor->GetOutput(), parser); 
  }
  else
  {
    // Do 3D metamorphosis
    return Metamorphosis<Image3DType>(referenceImage, parser);
  }

} // end main


template<typename TImage>
int Metamorphosis(typename TImage::Pointer fixedImage, typename ParserType::Pointer parser)
{
  // Construct metamorphosis
  typedef TImage ImageType;
  typedef itk::MetamorphosisImageRegistrationMethodv4<ImageType,ImageType>  MetamorphosisType;
  typename MetamorphosisType::Pointer metamorphosis = MetamorphosisType::New();

  // Read input (moving) image
  string inputPath;
  parser->GetCommandLineArgument("--in",inputPath);

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

  // Set input (moving) image
  metamorphosis->SetMovingImage(movingImage); // I_0

  // Set reference (fixed) image
  metamorphosis->SetFixedImage(fixedImage);   // I_1

  // Set metamorphosis parameters 
  if(parser->ArgumentExists("--scale"))
  {
    float scale;
    parser->GetCommandLineArgument("--scale",scale);
    metamorphosis->SetScale(scale);
  }

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
      metamorphosis->UseBiasOff();
    }
    else
    {
      metamorphosis->UseBiasOn();
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

  if(parser->ArgumentExists("--epsilon"))
  {
    double learningRate;
    parser->GetCommandLineArgument("--epsilon", learningRate);
    metamorphosis->SetLearningRate(learningRate);
    //metamorphosis->SetMinimumLearningRate(learningRate/1000);
  }

  if(parser->ArgumentExists("--fraction"))
  {
    double minimumFractionInitialEnergy;
    parser->GetCommandLineArgument("--fraction",minimumFractionInitialEnergy);
    metamorphosis->SetMinimumFractionInitialEnergy(minimumFractionInitialEnergy);
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

  // Run metamorphosis 
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

  cout<<"E = "<<metamorphosis->GetEnergy()<<" ("<<metamorphosis->GetImageEnergy()/metamorphosis->GetInitialEnergy()*100<<"%)"<<endl;
  cout<<"Time = "<<clock.GetTotal()<<"s"<<" ("<<clock.GetTotal()/60<<"m)"<<endl;
 
  // Write output images 
  int returnValue = EXIT_SUCCESS;

  // Compute I_0 o \phi_{10}
  typedef typename MetamorphosisType::OutputTransformType TransformType;
  typename TransformType::Pointer transform = const_cast<TransformType*>(metamorphosis->GetOutput()->Get()); // \phi_{10}
  transform->SetNumberOfIntegrationSteps(metamorphosis->GetNumberOfTimeSteps()+2);
  transform->SetLowerTimeBound(1.0);
  transform->SetUpperTimeBound(0.0);
  transform->IntegrateVelocityField();

  typedef itk::ResampleImageFilter<ImageType,ImageType,typename TransformType::ScalarType>   OutputResamplerType;
  typename OutputResamplerType::Pointer outputResampler = OutputResamplerType::New();
  outputResampler->SetInput(movingImage);   // I_0
  outputResampler->SetTransform(transform); // \phi_{10}
  outputResampler->UseReferenceImageOn();
  outputResampler->SetReferenceImage(fixedImage);


  outputResampler->Update();
  writeImage<ImageType>(outputResampler->GetOutput(),"test.tif");

  // Compute I(1) = I_0 o \phi_{10} + B(1)
  typedef itk::AddImageFilter<ImageType,typename MetamorphosisType::BiasImageType,ImageType>  AdderType;
  typename AdderType::Pointer adder = AdderType::New();
  adder->SetInput1(outputResampler->GetOutput());  // I_0 o \phi_{10}
  adder->SetInput2(metamorphosis->GetBias());      // B(1)
  adder->Update();

  // Limit intensity of I(1) to intensity range of ouput image type
  itkStaticConstMacro(ImageDimension, unsigned int, ImageType::ImageDimension);
  typedef unsigned char   OutputPixelType;
  typedef itk::Image<OutputPixelType,ImageDimension>  OutputImageType;

  typedef itk::ClampImageFilter<ImageType, OutputImageType> ClamperType;
  typename ClamperType::Pointer clamper = ClamperType::New();
  clamper->SetInput(adder->GetOutput());
  clamper->SetBounds(itk::NumericTraits<OutputPixelType>::min(), itk::NumericTraits<OutputPixelType>::max());
  clamper->Update();

  // Write output image, I(1)
  string outputPath;
  parser->GetCommandLineArgument("--out",outputPath);

  typedef itk::ImageFileWriter<OutputImageType>  OutputWriterType;
  typename OutputWriterType::Pointer outputWriter = OutputWriterType::New();
  outputWriter->SetInput(clamper->GetOutput()); // I(1)
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

  // Write displacement field, \phi_{10}
  typedef typename TransformType::DisplacementFieldType  FieldType;
  typedef itk::ImageFileWriter<FieldType>    FieldWriterType;

  string fieldPath;
  parser->GetCommandLineArgument("--field",fieldPath);

  if(fieldPath != "")
  {
    transform->SetNumberOfIntegrationSteps(metamorphosis->GetNumberOfTimeSteps()+2);
    transform->SetLowerTimeBound(1.0);
    transform->SetUpperTimeBound(0.0);
    transform->IntegrateVelocityField();

    typename FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
    fieldWriter->SetInput(transform->GetDisplacementField()); // \phi_{10}
    fieldWriter->SetFileName(fieldPath);
    try
    {
      fieldWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write displacement field: "<<fieldPath<<endl;
      cerr<<exceptionObject<<endl;
      returnValue = EXIT_FAILURE;
    }
  }

  // Write inverse displacement field, \phi_{01}
  string inverseFieldPath;
  parser->GetCommandLineArgument("--invfield", inverseFieldPath);
  if(inverseFieldPath != "")
  {
    transform->SetNumberOfIntegrationSteps(metamorphosis->GetNumberOfTimeSteps()+2);
    transform->SetLowerTimeBound(0.0);
    transform->SetUpperTimeBound(1.0);
    transform->IntegrateVelocityField();

    typename FieldWriterType::Pointer inverseFieldWriter = FieldWriterType::New();
    inverseFieldWriter->SetInput(transform->GetDisplacementField()); // \phi_{10}
    inverseFieldWriter->SetFileName(inverseFieldPath);
    try
    {
      inverseFieldWriter->Update();
    }
    catch(itk::ExceptionObject& exceptionObject)
    {
      cerr<<"Error: Could not write inverse displacement field: "<<inverseFieldPath<<endl;
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

    typedef itk::CastImageFilter<typename MetamorphosisType::BiasImageType,FloatImageType>  CasterType;
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
  parser->GetCommandLineArgument("--grid", gridPath);

  if(gridPath != "")
  {
    // Generate grid
    typedef itk::BSplineKernelFunction<0>  KernelType;
    typename KernelType::Pointer kernelFunction = KernelType::New();
    unsigned int gridStep = 5; // Space in voxels between grid lines

    typedef itk::GridImageSource<ImageType> GridSourceType;
    typename GridSourceType::Pointer gridSource = GridSourceType::New();
    typename GridSourceType::ArrayType gridSpacing = fixedImage->GetSpacing()*gridStep;
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

    // Apply transform to grid
    transform->SetNumberOfIntegrationSteps(metamorphosis->GetNumberOfTimeSteps()+2);
    transform->SetLowerTimeBound(1.0);
    transform->SetUpperTimeBound(0.0);
    transform->IntegrateVelocityField();

    typedef itk::ResampleImageFilter<ImageType,OutputImageType,typename TransformType::ScalarType>   GridResamplerType;
    typename GridResamplerType::Pointer gridResampler = GridResamplerType::New();
    gridResampler->SetInput(gridSource->GetOutput());
    gridResampler->SetTransform(transform);
    gridResampler->UseReferenceImageOn();
    gridResampler->SetReferenceImage(fixedImage);

    // Write grid to file
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
      //ss<<std::setprecision(4);// << std::fixed;
      double imageEnergy = filter->GetImageEnergy();
      double imageEnergyPercent = imageEnergy/filter->GetInitialEnergy()*100;
      
      ss<<filter->GetCurrentIteration()<<".\t"<<filter->GetEnergy()<<", "<<filter->GetVelocityEnergy()<<", "<<filter->GetRateEnergy()<<", "<<imageEnergy<<" ("<<imageEnergyPercent<<"%), ";
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
