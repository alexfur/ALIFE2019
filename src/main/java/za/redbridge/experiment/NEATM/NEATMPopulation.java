package za.redbridge.experiment.NEATM;

import org.encog.ml.ea.species.BasicSpecies;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATInnovationList;

import java.util.Random;

import za.redbridge.experiment.HyperNEAT.HyperNEATCODEC;
import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.Main;
import za.redbridge.experiment.MultiObjective.FactorMultiObjectiveHyperNEATGenome;
import za.redbridge.experiment.MultiObjective.FactorMultiObjectiveNEATMGenome;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.NEATM.sensor.SensorType;
import za.redbridge.experiment.NEAT.NEATPopulation;

/**
 * Created by jamie on 2014/09/08.
 */
public class NEATMPopulation extends NEATPopulation
{

    private static final long serialVersionUID = -6647644833955733411L;
    private SensorMorphology sensorMorphology;

    /**
     * An empty constructor for serialization.
     */

    private boolean multiObjective;
    public NEATMPopulation()
    {

    }

    /**
     * Construct a starting NEATM population. This does not generate the initial
     * random population of genomes.
     *
     * @param outputCount    The output neuron count.
     * @param populationSize The population size.
     */
    public NEATMPopulation(int outputCount, int populationSize, boolean multiObjective, SensorMorphology sensorMorphology)
    {
        super(SensorType.values().length, outputCount, populationSize); // inputCount = SensorType.values().length
                                                                        // This is because initial population begins with one of each sensor type.
        this.multiObjective = multiObjective;

        this.sensorMorphology = sensorMorphology;
    }



    /**
     * Construct a starting HyperNEATM population. This does not generate the
     * initial random population of genomes.
     *
     * @param theSubstrate
     *            The substrate ID.
     * @param populationSize
     *            Number of CPPNs in each generation.
     */
    public NEATMPopulation(final Substrate theSubstrate, final int populationSize, boolean multiObjective, boolean evolvingMorph, SensorMorphology sensorMorphology)
    {
        super(theSubstrate, populationSize);
        setInputCount(4);
        if(evolvingMorph)
        {
            setOutputCount(6);
        }
        else{
            setOutputCount(2);
        }
        this.multiObjective = multiObjective;
        this.sensorMorphology = sensorMorphology;
    }




    @Override
    public void reset()
    {
        // create the genome factory
        if (isHyperNEAT())                                  // Just checks if NEAT(M)Population has a valid substrate.
        {
            if(Main.Args.evolvingMorphology)
                setCODEC(new HyperNEATMCODEC());
            else
                setCODEC(new HyperNEATCODEC(sensorMorphology));

            if(multiObjective)
            {
                setGenomeFactory(new FactorMultiObjectiveHyperNEATGenome());
            }
            else
            {
                setGenomeFactory(new za.redbridge.experiment.HyperNEATM.FactorHyperNEATGenome());
            }

        }
        else
        {
            setCODEC(new NEATMCODEC());
            if(multiObjective)
            {
                setGenomeFactory(new FactorMultiObjectiveNEATMGenome());
            }
            else
            {
                setGenomeFactory(new FactorNEATMGenome());
            }
        }


        // create the new genomes
        getSpecies().clear();

        // reset counters
        getGeneIDGenerate().setCurrentID(1);
        getInnovationIDGenerate().setCurrentID(1);

        final Random rnd = getRandomNumberFactory().factor();

        // create one default species
        final BasicSpecies defaultSpecies = new BasicSpecies();
        defaultSpecies.setPopulation(this);

        // create the initial population
        for (int i = 0; i < getPopulationSize(); i++)
        {
            final NEATGenome genome = getGenomeFactory().factor(rnd, this,
                    getInputCount(), getOutputCount(), getInitialConnectionDensity());
            defaultSpecies.add(genome);
        }
        defaultSpecies.setLeader(defaultSpecies.getMembers().get(0));
        getSpecies().add(defaultSpecies);

        // create initial innovations
        setInnovations(new NEATInnovationList(this));
    }

}
