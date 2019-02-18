package za.redbridge.experiment;

import za.redbridge.experiment.NEATM.sensor.SensorModel;
import za.redbridge.experiment.NEATM.sensor.SensorMorphology;

import za.redbridge.simulator.khepera.*;

//import za.redbridge.experiment.NEATM.sensor.parameter.spec.ParameterType.*;
import static za.redbridge.experiment.NEATM.sensor.SensorType.*;


/**
 * A horrible adapter class for different representations of morphologies. Creates a
 * {@link SensorMorphology} for a {@link KheperaIIIPhenotype.Configuration}.
 * <p>
 * Created by jamie on 2014/10/06.
 */
public class KheperaIIIMorphology extends SensorMorphology
{

    private static final long serialVersionUID = 8121207679231125300L;

    private static final KheperaIIIPhenotype.Configuration DEFAULT_CONFIGURATION =
            new KheperaIIIPhenotype.Configuration();

    static
    {
        DEFAULT_CONFIGURATION.hyperNEATConfiguration = true;
    }

    public KheperaIIIMorphology()
    {
        this(DEFAULT_CONFIGURATION);
    }

    public KheperaIIIMorphology(KheperaIIIPhenotype.Configuration config)
    {
        super(createSensorModels(config));
    }

    private static SensorModel[] createSensorModels(KheperaIIIPhenotype.Configuration config)
    {
        int sensorIndex = 0;
        final int sensorCount = config.getNumberOfSensors();
        SensorModel[] sensorModels = new SensorModel[sensorCount];

        if (config.hyperNEATConfiguration)
        {
            sensorModels[sensorIndex++] = new SensorModel(BOTTOM_PROXIMITY);
            //low res cam
            sensorModels[sensorIndex++] = new SensorModel(LOW_RES_CAM, (float) Math.toRadians(0),0, LowResCameraSensor.RANGE, LowResCameraSensor.FOV);
            //cp
            sensorModels[sensorIndex++] = new SensorModel(COLOUR_PROXIMITY, (float) Math.toRadians(30),0, ColourProximitySensor.RANGE, ColourProximitySensor.FIELD_OF_VIEW);
            sensorModels[sensorIndex++] = new SensorModel(COLOUR_PROXIMITY, (float) Math.toRadians(-30),0, ColourProximitySensor.RANGE, ColourProximitySensor.FIELD_OF_VIEW);
            //ultrasonic
            sensorModels[sensorIndex++] = new SensorModel(ULTRASONIC, (float) Math.toRadians(60),0, UltrasonicSensor.RANGE, UltrasonicSensor.FIELD_OF_VIEW);
            sensorModels[sensorIndex++] = new SensorModel(ULTRASONIC, (float) Math.toRadians(-60),0, UltrasonicSensor.RANGE, UltrasonicSensor.FIELD_OF_VIEW);
            // proximity
            sensorModels[sensorIndex++] = new SensorModel(PROXIMITY, (float) Math.toRadians(120),0, ProximitySensor.RANGE,ProximitySensor.FIELD_OF_VIEW);
            sensorModels[sensorIndex++] = new SensorModel(PROXIMITY, (float) Math.toRadians(-120),0, ProximitySensor.RANGE, ProximitySensor.FIELD_OF_VIEW);
        }



        return sensorModels;
    }
}
