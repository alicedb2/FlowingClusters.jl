ebird_df = DataFrame(CSV.File("data/ebird_data/ebird_bioclim_landcover.csv", delim="\t"))
_left, _right, _bottom, _top = (1.1 * minimum(ebird_df.lon), 
                            0.9 * maximum(ebird_df.lon),
                            0.9 * minimum(ebird_df.lat), 
                            1.1 * maximum(ebird_df.lat))
lon_0, lat_0 = (_left + _right)/2, (_top + _bottom)/2
_left, _right, _bottom, _top = max(_left, -180.0), min(_right, 180.0), max(_bottom, -90.0), min(_top, 90.0)

eb_subsample = reduce(push!, sample(eachrow(ebird_df), 3000, replace=false), init=DataFrame())
# eb_subsample = ebird_df
eb_subsample_presences = subset(eb_subsample, :sp1 => x -> x .== 1.0)
eb_subsample_absences = subset(eb_subsample, :sp1 => x -> x .== 0.0)

# bioclim_layernames = "BIO" .* string.(1:19)

# Temperature, Annual precipitations
_layernames = "BIO" .* string.([1, 12])

# Temperature, Isothermality, Annual Precipitation
# _layernames = "BIO" .* string.([1, 3, 12])

# Temperature, Mean Diurnal Range, Isothermality, Temperature Seasonality, Annual Precipitation, Precipitation Seasonality
# _layernames = "BIO" .* string.([1, 2, 3, 4, 12, 15])

_layers = [SimpleSDMPredictor(RasterData(WorldClim2, BioClim), 
                                     layer=layer, resolution=10.0,
                                     left=_left, right=_right, bottom=_bottom, top=_top)
                 for layer in _layernames]

dataset = MNCRPDataset(eb_subsample_presences, _layers, longlatcols=["lon", "lat"], layernames=_layernames)
abs_dataset = MNCRPDataset(eb_subsample_absences, _layers, longlatcols=["lon", "lat"], layernames=_layernames)

train_presences, validation_presences, test_presences = split(dataset, 3)
train_absences, validation_absences, test_absences = split(abs_dataset, 3)

standardize!(train_presences)
standardize!(validation_presences, with=train_presences)
standardize!(test_presences, with=train_presences)
standardize!(train_absences, with=train_presences)
standardize!(validation_absences, with=train_presences)
standardize!(test_absences, with=train_presences)