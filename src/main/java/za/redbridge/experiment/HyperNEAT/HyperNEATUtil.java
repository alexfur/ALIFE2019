package za.redbridge.experiment.HyperNEAT;

import org.encog.ml.CalculateScore;
import org.encog.ml.ea.opp.CompoundOperator;
import org.encog.ml.ea.opp.selection.TruncationSelection;
import org.encog.neural.hyperneat.HyperNEATCODEC;
import org.encog.neural.neat.NEATCODEC;
import org.encog.neural.neat.training.opp.*;
import org.encog.neural.neat.training.opp.links.MutatePerturbLinkWeight;
import org.encog.neural.neat.training.opp.links.MutateResetLinkWeight;
import org.encog.neural.neat.training.opp.links.SelectFixed;
import org.encog.neural.neat.training.opp.links.SelectProportion;
import org.encog.neural.neat.training.species.OriginalNEATSpeciation;
import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.SingleObjective.SingleObjectiveEA;

/**
 * HyperNEATUtil for regular HyperNEAT network.
 *
 * Created by jamie on 2019/01/29
 */
public class HyperNEATUtil {

    public static SingleObjectiveEA constructNEATTrainer(
            final CalculateScore calculateScore, final int inputCount,
            final int outputCount, final int populationSize) {
        final org.encog.neural.neat.NEATPopulation pop = new org.encog.neural.neat.NEATPopulation(inputCount, outputCount,
                populationSize);
        pop.reset();
        return constructNEATTrainer(pop, calculateScore);
    }

    /**
     * Construct a NEAT (or HyperNEAT trainer.
     * @param population The population.
     * @param calculateScore The score function.
     * @return The NEAT EA trainer.
     */
    public static SingleObjectiveEA constructNEATTrainer(final org.encog.neural.neat.NEATPopulation population,
                                                         final CalculateScore calculateScore) {
        final SingleObjectiveEA result = new SingleObjectiveEA(population, calculateScore);
        result.setSpeciation(new OriginalNEATSpeciation());

        result.setSelection(new TruncationSelection(result, 0.3));
        final CompoundOperator weightMutation = new CompoundOperator();
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(1),
                        new MutatePerturbLinkWeight(0.02)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(2),
                        new MutatePerturbLinkWeight(0.02)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(3),
                        new MutatePerturbLinkWeight(0.02)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectProportion(0.02),
                        new MutatePerturbLinkWeight(0.02)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(1),
                        new MutatePerturbLinkWeight(1)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(2),
                        new MutatePerturbLinkWeight(1)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectFixed(3),
                        new MutatePerturbLinkWeight(1)));
        weightMutation.getComponents().add(
                0.1125,
                new NEATMutateWeights(new SelectProportion(0.02),
                        new MutatePerturbLinkWeight(1)));
        weightMutation.getComponents().add(
                0.03,
                new NEATMutateWeights(new SelectFixed(1),
                        new MutateResetLinkWeight()));
        weightMutation.getComponents().add(
                0.03,
                new NEATMutateWeights(new SelectFixed(2),
                        new MutateResetLinkWeight()));
        weightMutation.getComponents().add(
                0.03,
                new NEATMutateWeights(new SelectFixed(3),
                        new MutateResetLinkWeight()));
        weightMutation.getComponents().add(
                0.01,
                new NEATMutateWeights(new SelectProportion(0.02),
                        new MutateResetLinkWeight()));
        weightMutation.getComponents().finalizeStructure();

        result.setChampMutation(weightMutation);
        result.addOperation(0.5, new NEATCrossover());
        result.addOperation(0.494, weightMutation);
        result.addOperation(0.0005, new NEATMutateAddNode());
        result.addOperation(0.005, new NEATMutateAddLink());
        result.addOperation(0.0005, new NEATMutateRemoveLink());
        result.getOperators().finalizeStructure();

        if (population.isHyperNEAT()) {
            result.setCODEC(new HyperNEATCODEC());
        } else {
            result.setCODEC(new NEATCODEC());
        }

        return result;
    }
}

