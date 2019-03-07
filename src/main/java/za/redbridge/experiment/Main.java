package za.redbridge.experiment;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.encog.Encog;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.hyperneat.substrate.Substrate;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.training.NEATGenome;
import org.encog.neural.neat.training.NEATLinkGene;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import za.redbridge.experiment.HyperNEAT.HyperNEATCODEC;
import za.redbridge.experiment.HyperNEAT.HyperNEATUtil;
import za.redbridge.experiment.HyperNEATM.HyperNEATMCODEC;
import za.redbridge.experiment.HyperNEATM.HyperNEATMUtil;
import za.redbridge.experiment.HyperNEATM.SubstrateFactory;
import za.redbridge.experiment.MultiObjective.MultiObjectiveEA;
import za.redbridge.experiment.MultiObjective.MultiObjectiveHyperNEATUtil;
import za.redbridge.experiment.MultiObjective.MultiObjectiveNEATMUtil;
import za.redbridge.experiment.NEAT.NEATUtil;
import za.redbridge.experiment.NEATM.NEATMPopulation;
import za.redbridge.experiment.NEATM.NEATMUtil;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;
import za.redbridge.experiment.SingleObjective.SingleObjectiveEA;
import za.redbridge.simulator.config.SimConfig;

import java.io.*;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;

import static za.redbridge.experiment.Utils.isBlank;
import static za.redbridge.experiment.Utils.readObjectFromFile;

/**
 * Entry point for the experiment platform.
 * <p>
 * Created by jamie on 2014/09/09.
 */
public class Main
{
    private static final double CONVERGENCE_SCORE = 110;
    private static double SUBSTRATE_RADIUS = 1;

    public static void main(String[] args) throws IOException
    {
        try
        {
            BufferedWriter w = new BufferedWriter(new FileWriter("results/extracted_neural.csv", true));
            w.write("Generation,Average Neural,Max Neural");
            w.newLine();

            ArrayList<Double> numLinksList = new ArrayList<>();


            File file = new File("./results/");
            String[] dirsInResults = file.list(new FilenameFilter() {
                @Override
                public boolean accept(File current, String name) {
                    return new File(current, name).isDirectory();
                }
            });


            String pathPopulations = "./results/" + dirsInResults[0] + "/populations/";


            File file2 = new File(pathPopulations);
            String[] epochSersInPopulationsDir = file2.list(new FilenameFilter() {
                @Override
                public boolean accept(File current, String name) {
                    return new File(current, name).isFile();
                }
            });

            int epoch=1;

            for (String epochX : epochSersInPopulationsDir)
            {
                org.encog.neural.neat.NEATPopulation pop = (NEATPopulation) Utils.readObjectFromFile(pathPopulations + "/epoch-" + epoch + ".ser");

                for(Genome genome : pop.flatten())
                {
                    int countLinks =0;
                    for(NEATLinkGene link : ((NEATGenome)genome).getLinksChromosome())
                    {
                        if(link.isEnabled())
                        {
                            countLinks++;
                        }
                    }
                    numLinksList.add((100-countLinks)/100.0);
                }

                double avNumLinks = 0;
                for (int i = 0; i < numLinksList.size(); i++)
                {
                    avNumLinks += numLinksList.get(i);
                }
                avNumLinks = avNumLinks / numLinksList.size();
                double maxNumLinks = Collections.max(numLinksList);

                w.write(epoch+","+avNumLinks+","+maxNumLinks);
                w.newLine();

                System.out.println("av num links: " + avNumLinks);
                System.out.println("max num links: " + maxNumLinks);

                epoch++;
            }

            w.close();




        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static class Args
    {
        @Parameter(names = "-c", description = "Simulation config file to load")
        public static String configFile = "config/bossConfig.yml";
        //public static String configFile = "config/ConfigSimple.yml";
        //public static String configFile = "config/ConfigMedium.yml";
        // public static String configFile = "config/ConfigDifficult.yml";

        @Parameter(names = "-g", description = "Number of generations to train for")    // Jamie calls this 'iterations'
        public static int numGenerations = 3;

        @Parameter(names = "-p", description = "Initial population size")
        public static int populationSize = 3;

        @Parameter(names = "--trials", description = "Number of simulation runs per iteration (team lifetime)")
        // Jamie calls this 'simulationRuns' (and 'lifetime' in his paper)
        public static int trialsPerIndividual = 1;

        @Parameter(names = "--conn-density", description = "Adjust the initial connection density"
                + " for the population")
        //NEAT
        //public static double connectionDensity = 0.5;
        //HyperNEAT
        private double connectionDensity = 0.8;

        @Parameter(names = "--demo", description = "Show a GUI demo of a given genome")
        public static String genomePath = null;

        @Parameter(names = "--evolvingMorph", description = "Evolving morphology")
        public static boolean evolvingMorphology = false;

        @Parameter(names = "--HyperNEATM", description = "Using HyperNEATM")
        public static boolean hyperNEATM = true;

        @Parameter(names = "--population", description = "To resume a previous experiment, provide"
                + " the path to a serialized population")
        public static String populationPath =null;

        //private String populationPath = "/mnt/lustre/users/dnagar/experiment-...";

        //private String populationPath = "/mnt/lustre/users/afurman/experiment-...";

        @Parameter(names = "--threads", description = "Number of threads to run simulations with."
                + " By default Runtime#availableProcessors() is used to determine the number of threads to use")
        public static int threads = 0;

        @Parameter(names = "--multi-objective", description = "Using Multi-Objective NEAT/HyperNEAT"
                +" Based on NEAT-MODS")
        public static boolean multiObjective = false;

        @Override
        public String toString()
        {
            return "Options: \n"
                    + "\tConfig file path: " + configFile + "\n"
                    + "\tNumber of simulation steps: " + numGenerations + "\n"
                    + "\tPopulation size: " + populationSize + "\n"
                    + "\tNumber of simulation tests per iteration: " + trialsPerIndividual + "\n"
                    + "\tInitial connection density: " + connectionDensity + "\n"
                    + "\tDemo network config path: " + genomePath + "\n"
                    + "\tEvolving morphology: " + evolvingMorphology + "\n"
                    + "\tHyperNEATM: " + hyperNEATM + "\n"
                    + "\tPopulation path: " + populationPath + "\n"
                    + "\tNumber of threads: " + threads + "\n"
                    + "\tMulti-objective: " + multiObjective;
        }
    }
}
