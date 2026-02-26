import ee

def get_gee_data():
    """
    This function executes the Earth Engine machine learning pipeline
    and returns the processed Feature Collections for Streamlit.
    """
    
    # 1. Load Bengaluru lakes
    lakes = ee.FeatureCollection('projects/gee-lake-project/assets/bengaluru_lakes')

    # Create 500m and 1000m buffers
    lakes_500m = lakes.map(lambda feature: feature.buffer(500))
    lakes_1000m = lakes.map(lambda feature: feature.buffer(1000))

    # 2. Setup Machine Learning Bands
    bands = ['B2','B3','B4','B8','B11','B12','NDVI','NDBI']
    baseBands = ['B2','B3','B4','B8','B11','B12']

    # Sentinel-2 composite for ML (UPDATED TO HARMONIZED)
    s2_ml = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(lakes_1000m) \
        .filterDate('2024-01-01', '2024-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
        .select(baseBands) \
        .median()

    # Add NDVI & NDBI
    ndvi = s2_ml.normalizedDifference(['B8','B4']).rename('NDVI')
    ndbi = s2_ml.normalizedDifference(['B11','B8']).rename('NDBI')
    s2_enhanced = s2_ml.addBands([ndvi, ndbi])

    # 3. Training Data Geometries
    geometry_coords = [
        [77.67212085942283, 12.937094514889418], [77.65774421910301, 12.933121013187428], [77.7415226318519, 12.951551110151945], [77.61787424413254, 12.90982460438119], [77.62043198727666, 12.981957944520174], [77.7300777035543, 13.024010686135421], [77.72853275116172, 13.024470610860686], [77.74076299346838, 13.043737794851372], [77.74072007812414, 13.043387652225661], [77.66149024242759, 13.047641714485033], [77.66237000698447, 13.046805565369818], [77.66795973057151, 13.046951891669098], [77.68959110692406, 13.140048303530696], [77.69011681989097, 13.139421427156583], [77.69112533048057, 13.140487116039905], [77.69069617703819, 13.141113989692078], [77.69102877095604, 13.139588594346229], [77.74932058824893, 13.14661827156793], [77.74921329988834, 13.145260073765415], [77.74889143480655, 13.144110823598036], [77.75138052477237, 13.145489923152889], [77.74545820726749, 13.14858242124288], [77.74506347066057, 13.149353093820748], [77.44235259446201, 13.21883309596088], [77.44282466324863, 13.215449034211991], [77.44488459977207, 13.213819654406699], [77.44436961564121, 13.218415313099829], [77.44106513413486, 13.21486412989275], [77.52833209261004, 13.231146289127985], [77.52726993784015, 13.231083624909353], [77.56844680638785, 13.275131409961995], [77.56999175878043, 13.29835363361134], [77.56462723535296, 13.311112266587662]
    ]
    geometry2_coords = [
        [77.60985036596452, 12.983040771391789], [77.6109447072426, 12.98253895149632], [77.59379330205262, 12.989927912292568], [77.59285587000191, 12.989709679085191], [77.72361595940515, 13.027945051115987], [77.62309833719735, 12.90659943858868], [77.62351676180367, 12.906128838176544], [77.62257262423043, 12.90566869469494], [77.62196108057503, 12.908858308236956], [77.62627407267098, 12.9068713406453], [77.62370988085274, 12.903211095853495], [77.6228214464585, 12.900027997089051], [77.62068104366462, 12.898674982968437], [77.62077357987563, 12.898677597495539], [77.62075480441253, 12.898578245446688], [77.62068774918716, 12.898328557886584], [77.62057509640853, 12.898322021559677], [77.62048390130202, 12.89831940702887], [77.6204087994496, 12.89823704929431], [77.6204191162337, 12.89835161728361], [77.62032322726142, 12.8984182878022], [77.62032322726142, 12.89834900275311], [77.62038526730228, 12.89806629605758], [77.62031150655437, 12.898057145190274], [77.55401552119984, 12.943797678186083], [77.5543682316853, 12.943691808945609], [77.5543682316853, 12.943596395887907], [77.5541697482182, 12.943429096191734], [77.5541697482182, 12.943521895255827], [77.55418450036778, 12.943623843483595], [77.55426228442921, 12.94361730834204], [77.55426898995175, 12.943725791669685], [77.55417913594975, 12.943717949502998], [77.55414963165059, 12.94378330088445], [77.55409464636578, 12.94378330088445], [77.55406916538014, 12.943680045693903], [77.5538988451077, 12.943702265168904], [77.55381301441922, 12.943716642475195], [77.55380362668767, 12.943804213322908], [77.55381569662823, 12.9438957052205], [77.5538103322102, 12.943976740873186], [77.5538264254643, 12.94407868891494], [77.53424217113889, 12.934797110562641], [77.53469278225339, 12.934556608728368], [77.53057495488494, 12.997347321604533], [77.53048912419646, 12.997415272389931], [77.53043011559814, 12.997423112863963], [77.53041670455306, 12.997224487445294], [77.53052139528634, 12.997303546582245], [77.53054888792875, 12.997289172372728], [77.53060588487031, 12.99727675828202]
    ]
    geometry3_coords = [
        [77.60375152998161, 12.954232119674769], [77.59142555685516, 12.973049891452446], [77.59191371889587, 12.972825109049827], [77.5912887641954, 12.972720559025893], [77.59162135811324, 12.97260294019646], [77.49568237161226, 12.98214024631636], [77.49592913484163, 12.983122977570718], [77.4963295944153, 12.983252292490363], [77.49639396743166, 12.98317780372707], [77.49657635764467, 12.983173883265222], [77.49575348218629, 13.295850898209954], [77.49616117795655, 13.295788250670123], [77.49650450071046, 13.295746485634579], [77.49995918592164, 13.296143253181773], [77.50038833936402, 13.296205900629865], [77.49976606687257, 13.299129430202237], [77.50189037641236, 13.29915031242951], [77.50066728910157, 13.298941490075869], [77.50191183408448, 13.29338674944077], [77.50068874677369, 13.292676735826436], [77.50513789132124, 13.291970411331624], [77.4945592589665, 13.293745446639592], [77.49009606316572, 13.29464340072886], [77.4910831160832, 13.29464340072886], [77.48981711342817, 13.29149010542675], [77.48773571923262, 13.294121334802536], [77.48506429634818, 13.297337243043986], [77.48547199211845, 13.29656459365828], [77.48585823021659, 13.294016921482378], [77.4860942646099, 13.292910137524823], [77.48536470375785, 13.292931020287753], [77.48469951592216, 13.292931020287753], [77.48291852913627, 13.293118965073253], [77.48193147621879, 13.292889254760102], [77.48169544182548, 13.29199129417548], [77.47785451851615, 13.292492481888507], [77.47729661904106, 13.292033059857797], [77.47832658730277, 13.290863618034434], [77.47909906349906, 13.290696554456522], [77.47909906349906, 13.289944766931981], [77.48023632012138, 13.289297392474335], [77.47997882805595, 13.288441184892978], [77.59407860177535, 13.397032965472068], [77.59294711760054, 13.395698206725106], [77.59278618505965, 13.394591888587112], [77.59164892843734, 13.396000877782232]
    ]
    geometry4_coords = [
        [77.66055181751513, 12.935169075595402], [77.663770468333, 12.934583506380068], [77.66389921436571, 12.938682462020543], [77.65849188099169, 12.929940730385162], [77.65252664814257, 12.929982557581528], [77.64656141529345, 12.93249217653421], [77.64553144703173, 12.931530158918823], [77.6455510764758, 12.931583771211507], [77.65659135752037, 12.926214169406673], [77.66152671374516, 12.925419953043072], [77.64942458666997, 12.929058958814975], [77.64522277132335, 12.93141278234762], [77.64698230043712, 12.931391868870177], [77.64743291155162, 12.928903152539949], [77.64787279383006, 12.92807706054102], [77.64787279383006, 12.927794724673992], [77.64290534273448, 12.932489824066737], [77.6415749670631, 12.931193190746994], [77.64206849352183, 12.933556404488543], [77.64949284807506, 12.939537535698317], [77.66122226944223, 12.935215965314866], [77.66175871124521, 12.936909925343913], [77.66231661072031, 12.937119055413904], [77.66250972976938, 12.938352919257675], [77.66315345993296, 12.93885482822721], [77.66476278534189, 12.941448008477332], [77.6686895393397, 12.939398561804332], [77.66486057893175, 12.928293104533221], [77.66140471977614, 12.92425340768702], [77.66187678856276, 12.925591904990595], [77.65981685203933, 12.920739818127217], [77.66033183617019, 12.920405187971859]
    ]

    waterPoints = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c), {'class': 0}) for c in geometry_coords])
    builtupPoints = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c), {'class': 1}) for c in geometry2_coords])
    vegetationPoints = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c), {'class': 2}) for c in geometry3_coords])
    barrenPoints = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(c), {'class': 3}) for c in geometry4_coords])

    trainingPoints = waterPoints.merge(builtupPoints).merge(vegetationPoints).merge(barrenPoints)

    # 4. Train Random Forest Model
    trainingData = s2_enhanced.select(bands).sampleRegions(
        collection=trainingPoints,
        properties=['class'],
        scale=10
    )

    withRandom = trainingData.randomColumn('random')
    trainSet = withRandom.filter(ee.Filter.lt('random', 0.7))
    testSet = withRandom.filter(ee.Filter.gte('random', 0.7))

    rfClassifier = ee.Classifier.smileRandomForest(200).train(
        features=trainSet,
        classProperty='class',
        inputProperties=bands
    )

    classified = s2_enhanced.select(bands).classify(rfClassifier)
    builtupMask = classified.eq(1)

    # 5. Core Processing Functions 
    def getBuiltupPercentage(lakeFeature):
        buffer500 = lakeFeature.geometry().buffer(500)
        builtupArea = builtupMask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer500,
            scale=10,
            maxPixels=1e9
        ).get('classification')

        builtupArea = ee.Number(ee.Algorithms.If(builtupArea, builtupArea, 0))

        totalArea = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=buffer500,
            scale=10,
            maxPixels=1e9
        ).values().get(0)

        return ee.Number(builtupArea).divide(ee.Number(totalArea)).multiply(100)

    def getWaterStress(lakeFeature):
        geom = lakeFeature.geometry()
        
        def map_ndwi(img):
            return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

        # UPDATED TO HARMONIZED and added early select
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate('2024-01-01', '2024-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B3', 'B8']) \
            .map(map_ndwi)

        ndwiMedian = s2.median().clip(geom)
        waterMask = ndwiMedian.gt(0)

        waterArea = waterMask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=10,
            maxPixels=1e9
        ).values().get(0)

        stress = ee.Number(1).subtract(
            ee.Number(waterArea).divide(ee.Number(geom.area()))
        ).multiply(100)

        return stress

    def getYearWaterArea(lakeFeature, startDate, endDate):
        geom = lakeFeature.geometry()
        # UPDATED TO HARMONIZED
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate(startDate, endDate) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B3','B8'])

        count = s2.size()

        def ndwi_map(img):
            return img.normalizedDifference(['B3','B8']).rename('NDWI')

        waterArea = ee.Algorithms.If(
            count.gt(0),
            s2.map(ndwi_map).median().gt(0).clip(geom).multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geom,
                scale=10,
                maxPixels=1e9
            ).get('NDWI'),
            0
        )
        return ee.Number(waterArea).divide(1e6)

    # 6. Multi-Year and Monthly Reductions
    years = ee.List.sequence(2018, 2024)
    months = ee.List.sequence(1, 12)

    def map_lake_year(lake):
        lakeName = lake.get('name')
        def map_year(year):
            year = ee.Number(year)
            start = ee.Date.fromYMD(year, 1, 1)
            end = start.advance(1, 'year')
            waterArea = getYearWaterArea(lake, start.format('YYYY-MM-dd'), end.format('YYYY-MM-dd'))
            return ee.Feature(None, {'Lake': lakeName, 'Year': year, 'Water_Area_sqkm': waterArea})
        return ee.FeatureCollection(years.map(map_year))
    
    yearlyWaterStats = lakes.map(map_lake_year).flatten()

    def map_lake_month(lake):
        lakeName = lake.get('name')
        def map_year(year):
            year = ee.Number(year)
            def map_month(month):
                month = ee.Number(month)
                start = ee.Date.fromYMD(year, month, 1)
                end = start.advance(1, 'month')
                waterArea = getYearWaterArea(lake, start.format('YYYY-MM-dd'), end.format('YYYY-MM-dd'))
                return ee.Feature(None, {'Lake': lakeName, 'Year': year, 'Month': month, 'Water_Area_sqkm': waterArea})
            return ee.FeatureCollection(months.map(map_month))
        return ee.FeatureCollection(years.map(map_year)).flatten()
        
    monthlyWaterStats = lakes.map(map_lake_month).flatten()

    # 7. Compute Risk Scores, Water Quality, and Rank
    def map_risk(lake):
        lakeName = lake.get('name')
        geom = lake.geometry()
        encroachment = getBuiltupPercentage(lake)
        waterStress = getWaterStress(lake)
        
        # Historical water comparison
        water2020 = getYearWaterArea(lake, '2020-01-01', '2020-12-31')
        water2024 = getYearWaterArea(lake, '2024-01-01', '2024-12-31')
        
        changePercent = ee.Algorithms.If(
            water2020.gt(0),
            ee.Number(water2024).subtract(water2020).divide(water2020).multiply(100),
            0
        )
        waterLossScore = ee.Number(changePercent).lt(0).multiply(ee.Number(changePercent).abs())
        
        # --- NEW: WATER QUALITY MODULE (NDCI & NDTI) ---
        # UPDATED TO HARMONIZED and added explicit select()
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geom) \
            .filterDate('2024-01-01', '2024-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .select(['B3', 'B4', 'B5', 'B8'])
            
        # Create a water mask so we only test the actual water for pollution
        ndwi_median = s2.map(lambda img: img.normalizedDifference(['B3', 'B8']).rename('NDWI')).median().clip(geom)
        waterMask = ndwi_median.gt(0)
        
        s2_water_only = s2.median().updateMask(waterMask).clip(geom)
        
        # Calculate Algae (NDCI) and Turbidity (NDTI)
        ndci = s2_water_only.normalizedDifference(['B5', 'B4']).rename('NDCI')
        ndti = s2_water_only.normalizedDifference(['B4', 'B3']).rename('NDTI')
        
        avg_ndci = ndci.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=10, maxPixels=1e9).get('NDCI')
        avg_ndti = ndti.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=10, maxPixels=1e9).get('NDTI')
        
        # Handle nulls if lake is totally dry
        avg_ndci = ee.Number(ee.Algorithms.If(avg_ndci, avg_ndci, 0))
        avg_ndti = ee.Number(ee.Algorithms.If(avg_ndti, avg_ndti, 0))
        
        # Normalize water quality scores (0 to 100 scale)
        algae_score = avg_ndci.add(1).divide(2).multiply(100)
        turbidity_score = avg_ndti.add(1).divide(2).multiply(100)
        
        # Combined Pollution Penalty
        pollution_penalty = algae_score.add(turbidity_score).multiply(0.25)
        
        # 40% Encroachment, 20% Water Stress, 20% Water Loss, 20% Pollution
        riskScore = ee.Number(encroachment).multiply(0.4) \
            .add(ee.Number(waterStress).multiply(0.2)) \
            .add(ee.Number(waterLossScore).multiply(0.2)) \
            .add(ee.Number(pollution_penalty).multiply(0.2))
            
        return ee.Feature(None, {
            'Lake': lakeName, 
            'Encroachment_Percent': encroachment, 
            'Water_Stress_Score': waterStress,
            'Water_2020_sqkm': water2020, 
            'Water_2024_sqkm': water2024,
            'Water_Change_Percent': changePercent, 
            'Algae_NDCI_Score': algae_score,
            'Turbidity_NDTI_Score': turbidity_score,
            'Risk_Score': riskScore
        })

    lakeRiskScores = lakes.map(map_risk)

    def map_category(f):
        score = ee.Number(f.get('Risk_Score'))
        category = ee.Algorithms.If(score.gte(70), 'Critical',
                     ee.Algorithms.If(score.gte(50), 'High',
                       ee.Algorithms.If(score.gte(30), 'Medium', 'Low')))
        return f.set('Risk_Category', category)

    lakeRiskWithCategory = lakeRiskScores.map(map_category)

    # Sorting and Indexing
    rankedLakes = lakeRiskWithCategory.sort('Risk_Score', False)
    rankedList = rankedLakes.toList(rankedLakes.size())
    indexed = ee.List.sequence(0, rankedList.size().subtract(1))

    def map_rank(i):
        feature = ee.Feature(rankedList.get(i))
        return feature.set('Rank', ee.Number(i).add(1)).set('Priority_Index', feature.get('Risk_Score'))

    finalWithRank = ee.FeatureCollection(indexed.map(map_rank))

    return finalWithRank, yearlyWaterStats, monthlyWaterStats