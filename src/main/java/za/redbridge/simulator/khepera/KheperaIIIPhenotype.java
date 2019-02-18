package za.redbridge.simulator.khepera;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import za.redbridge.simulator.phenotype.Phenotype;
import za.redbridge.simulator.sensor.AgentSensor;

/**
 * Phenotype that mimics the morphology of a Khepera III robot.
 * Created by jamie on 2014/09/22.
 */
public abstract class KheperaIIIPhenotype implements Phenotype {

    private final List<AgentSensor> sensors;

    private final Configuration configuration;

    /**
     * Constructor used for phenotypes that make use of {@link #configure(Map)}. Sensors will not
     * be initialized (i.e. there will be no sensors) if this constructor is used until the
     * phenotype is configured.
     * TODO: Implement configure
     */
    public KheperaIIIPhenotype() {
        configuration = new Configuration();
        sensors = new ArrayList<>();
    }

    /**
     * Default constructor for this phenotype. Sensors will be configured according to
     * configuration.
     */
    public KheperaIIIPhenotype(Configuration config) {
        configuration = new Configuration(config);
        sensors = new ArrayList<>(config.getNumberOfSensors());
        
        initSensors();
    }

    private void initSensors() {
        // Proximity sensors

        if(configuration.hyperNEATConfiguration){
            sensors.add(createLowResCameraSensor((float) Math.toRadians(0), 0f));
            //cp
            sensors.add(createColorPrixmitySensor((float) Math.toRadians(30), 0f));
            sensors.add(createColorPrixmitySensor((float) Math.toRadians(-30), 0f));
            //ultra
            sensors.add(createUltrasonicSensor((float) Math.toRadians(60), 0f));
            sensors.add(createUltrasonicSensor((float) Math.toRadians(-60), 0f));
            //proximity
            sensors.add(createProximitySensor((float) Math.toRadians(120), 0f));
            sensors.add(createProximitySensor((float) Math.toRadians(-120), 0f));
            //bottom proximity
            sensors.add(createBottomProximitySensor());
        }






    }

    /** Method can be overridden to customize proximity sensor */
    protected AgentSensor createProximitySensor(float bearing, float orientation) {
        return new ProximitySensor(bearing, orientation);
    }

    /** Method can be overridden to customize bottom proximity sensor */
    protected AgentSensor createBottomProximitySensor() {
        return new BottomProximitySensor();
    }

    /** Method can be overridden to customize ultrasonic sensor */
    protected AgentSensor createUltrasonicSensor(float bearing, float orientation) {
        return new UltrasonicSensor(bearing, orientation);
    }

    /** Method can be overridden to customize low res camera sensr */
    protected AgentSensor createLowResCameraSensor(float bearing, float orientation) {
        return new LowResCameraSensor(bearing, orientation);
    }

    protected AgentSensor createColorPrixmitySensor(float bearing, float orientation) {
        return new ColourProximitySensor(bearing, orientation);
    }

    /** Returns a copy of the current configuration */
    public Configuration getConfiguration() {
        return new Configuration(configuration);
    }

    @Override
    public List<AgentSensor> getSensors() {
        return sensors;
    }

    @Override
    public KheperaIIIPhenotype clone() throws UnsupportedOperationException {
        throw new UnsupportedOperationException(
                "This is an abstract class, implement clone in your subclass");
    }

    @Override
    public void configure(Map<String, Object> phenotypeConfigs) {
        // TODO: Implementation
        //initSensors();
    }

    /**
     * Configuration for the Khepera III phenotype that allows sensors to be enabled or disabled.
     * By default all sensors are disabled.
     */
    public static class Configuration {

        public boolean hyperNEATConfiguration = true;



        public Configuration() {
        }

        private Configuration(Configuration other) {
            this.hyperNEATConfiguration = other.hyperNEATConfiguration;
        }
        
        public int getNumberOfSensors() {
            int numSensors = 0;

            if (hyperNEATConfiguration) numSensors += 8;

            
            return numSensors;
        }
    }

}
