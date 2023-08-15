from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from sklearn.metrics import top_k_accuracy_score
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from keras import models
import numpy as np
from matplotlib import pyplot

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight
from keras.models import load_model

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 31620
nbr_validation_samples = 10667
nbr_epochs = 50
batch_size = 32

# Pre-train the whole network on the WildFish dataset

train_data_dir='C:\\Users\\bxiong\\Downloads\\WildFish\\WildFish_train\\WildFish_train'
val_data_dir='C:\\Users\\bxiong\\Downloads\\WildFish\\WildFish_validation\\WildFish_validation'
test_data_dir='C:\\Users\\bxiong\\Downloads\\WildFish\\WildFish_test\\WildFish_test'


FishNames = ['Abactochromis_labrosus', 'Abalistes_stellaris', 'Ablabys_taenianotus', 'Ablennes_hians', 'Abramis_brama', 'Abramites_hypselonotus', 'Abudefduf_bengalensis', 'Abudefduf_luridus', 'Abudefduf_saxatilis', 'Abudefduf_septemfasciatus', 'Abudefduf_sexfasciatus', 'Abudefduf_sordidus', 'Abudefduf_taurus', 'Abudefduf_vaigiensis', 'Acanthistius_ocellatus', 'Acanthochromis_polyacanthus', 'Acanthogobius_flavimanus', 'Acanthopagrus_bifasciatus', 'Acanthostracion_polygonius', 'Acanthostracion_quadricornis', 'Acanthurus_achilles', 'Acanthurus_bahianus', 'Acanthurus_chirurgus', 'Acanthurus_coeruleus', 'Acanthurus_dussumieri', 'Acanthurus_guttatus', 'Acanthurus_japonicus', 'Acanthurus_leucosternon', 'Acanthurus_lineatus', 'Acanthurus_nigricans', 'Acanthurus_nigricauda', 'Acanthurus_nigrofuscus', 'Acanthurus_olivaceus', 'Acanthurus_pyroferus', 'Acanthurus_sohal', 'Acanthurus_tennentii', 'Acanthurus_triostegus', 'Acanthurus_tristis', 'Acanthurus_xanthopterus', 'Acarichthys_heckelii', 'Acentrogobius_caninus', 'Achoerodus_gouldii', 'Acreichthys_radiatus', 'Acreichthys_tomentosus', 'Aeoliscus_strigatus', 'Aequidens_diadema', 'Aethaloperca_rogaa', 'Aetobatus_narinari', 'Aetobatus_ocellatus', 'Alectis_ciliaris', 'Alepes_vari', 'Alepisaurus_ferox', 'Allothunnus_fallai', 'Alopias_pelagicus', 'Aluterus_monoceros', 'Aluterus_scriptus', 'Amanses_scopas', 'Amatitlania_nigrofasciata', 'Amblyeleotris_diagonalis', 'Amblyeleotris_guttata', 'Amblyeleotris_periophthalma', 'Amblyeleotris_randalli', 'Amblyeleotris_wheeleri', 'Amblyglyphidodon_aureus', 'Amblyglyphidodon_curacao', 'Amblyglyphidodon_leucogaster', 'Amblygobius_decussatus', 'Amblygobius_nocturnus', 'Amblygobius_phalaena', 'Amblypomacentrus_breviceps', 'Amblyraja_hyperborea', 'Amphichaetodon_howensis', 'Amphiprion_akallopisos', 'Amphiprion_akindynos', 'Amphiprion_bicinctus', 'Amphiprion_chrysopterus', 'Amphiprion_clarkii', 'Amphiprion_ephippium', 'Amphiprion_frenatus', 'Amphiprion_latezonatus', 'Amphiprion_mccullochi', 'Amphiprion_melanopus', 'Amphiprion_nigripes', 'Amphiprion_ocellaris', 'Amphiprion_percula', 'Amphiprion_perideraion', 'Amphiprion_polymnus', 'Amphiprion_rubrocinctus', 'Amphiprion_sandaracinos', 'Anabas_testudineus', 'Anampses_caeruleopunctatus', 'Anampses_femininus', 'Anampses_lennardi', 'Anampses_melanurus', 'Anampses_meleagrides', 'Anampses_neoguinaicus', 'Anampses_twistii', 'Anisotremus_surinamensis', 'Anisotremus_virginicus', 'Anoplocapros_inermis', 'Anoplogaster_cornuta', 'Antennarius_coccineus', 'Antennarius_commerson', 'Antennarius_striatus', 'Antigonia_capros', 'Anyperodon_leucogrammicus', 'Aphareus_furca', 'Apistus_carinatus', 'Apogon_doryssa', 'Apolemichthys_trimaculatus', 'Aprion_virescens', 'Aptychotrema_rostrata', 'Aracana_aurita', 'Aracana_ornata', 'Archosargus_probatocephalus', 'Argyropelecus_aculeatus', 'Argyropelecus_gigas', 'Arothron_caeruleopunctatus', 'Arothron_hispidus', 'Arothron_immaculatus', 'Arothron_manilensis', 'Arothron_mappa', 'Arothron_meleagris', 'Arothron_nigropunctatus', 'Arothron_reticularis', 'Arothron_stellatus', 'Arripis_georgianus', 'Assessor_flavissimus', 'Asterropteryx_semipunctata', 'Atule_mate', 'Aulostomus_chinensis', 'Aulostomus_maculatus', 'Auxis_rochei', 'Balistapus_undulatus', 'Balistes_capriscus', 'Balistes_vetula', 'Balistoides_conspicillum', 'Balistoides_viridescens', 'Batrachomoeus_dubius', 'Beryx_splendens', 'Blenniella_chrysospilos', 'Bodianus_anthioides', 'Bodianus_axillaris', 'Bodianus_bilunulatus', 'Bodianus_diana', 'Bodianus_loxozonus', 'Bodianus_mesothorax', 'Bodianus_perditio', 'Bodianus_rufus', 'Bolbometopon_muricatum', 'Bothus_mancus', 'Bothus_ocellatus', 'Brachaelurus_waddi', 'Brachaluteres_jacksonianus', 'Brachionichthys_hirsutus', 'Brachysomophis_crocodilinus', 'Brama_brama', 'Bryaninops_erythrops', 'Caesio_caerulaurea', 'Caesio_cuning', 'Caesio_lunaris', 'Caesio_teres', 'Caesio_xanthonota', 'Caesioperca_lepidoptera', 'Calamus_bajonado', 'Calamus_calamus', 'Callechelys_marmorata', 'Callochromis_macrops', 'Callochromis_melanostigma', 'Callochromis_pleurospilus', 'Calloplesiops_altivelis', 'Callorhinchus_milii', 'Calotomus_carolinus', 'Calotomus_viridescens', 'Campostoma_anomalum', 'Cantherhines_dumerilii', 'Cantherhines_fronticinctus', 'Cantherhines_macrocerus', 'Cantherhines_pardalis', 'Cantherhines_pullus', 'Canthidermis_sufflamen', 'Canthigaster_amboinensis', 'Canthigaster_bennetti', 'Canthigaster_compressa', 'Canthigaster_epilampra', 'Canthigaster_jactator', 'Canthigaster_janthinoptera', 'Canthigaster_margaritata', 'Canthigaster_punctatissima', 'Canthigaster_rostrata', 'Canthigaster_smithae', 'Canthigaster_valentini', 'Carangoides_chrysophrys', 'Carangoides_coeruleopinnatus', 'Carangoides_fulvoguttatus', 'Caranx_crysos', 'Caranx_ignobilis', 'Caranx_latus', 'Caranx_lugubris', 'Caranx_melampygus', 'Caranx_ruber', 'Caranx_sexfasciatus', 'Carassius_auratus', 'Carassius_carassius', 'Carassius_gibelio', 'Carcharhinus_albimarginatus', 'Carcharhinus_amblyrhynchos', 'Carcharhinus_brachyurus', 'Carcharhinus_falciformis', 'Carcharhinus_galapagensis', 'Carcharhinus_leucas', 'Carcharhinus_limbatus', 'Carcharhinus_longimanus', 'Carcharhinus_melanopterus', 'Carcharias_taurus', 'Carcharodon_carcharias', 'Carinotetraodon_lorteti', 'Carinotetraodon_travancoricus', 'Carnegiella_strigata', 'Catlocarpio_siamensis', 'Centroberyx_affinis', 'Centrolophus_niger', 'Centropogon_australis', 'Centropristis_ocyurus', 'Centropristis_striata', 'Centropyge_abei', 'Centropyge_aurantia', 'Centropyge_bicolor', 'Centropyge_bispinosa', 'Centropyge_boylei', 'Centropyge_eibli', 'Centropyge_ferrugata', 'Centropyge_fisheri', 'Centropyge_flavissima', 'Centropyge_heraldi', 'Centropyge_loriculus', 'Centropyge_multicolor', 'Centropyge_multispinis', 'Centropyge_nox', 'Centropyge_potteri', 'Centropyge_shepardi', 'Centropyge_tibicen', 'Centropyge_venusta', 'Centropyge_vrolikii', 'Cephalopholis_argus', 'Cephalopholis_boenak', 'Cephalopholis_cruentata', 'Cephalopholis_cyanostigma', 'Cephalopholis_formosa', 'Cephalopholis_fulva', 'Cephalopholis_hemistiktos', 'Cephalopholis_leopardus', 'Cephalopholis_miniata', 'Cephalopholis_sexmaculata', 'Cephalopholis_sonnerati', 'Cephalopholis_spiloparaea', 'Cephalopholis_urodeta', 'Cetorhinus_maximus', 'Cetoscarus_bicolor', 'Chaenopsis_ocellata', 'Chaetodermis_penicilligerus', 'Chaetodipterus_faber', 'Chaetodon_adiergastos', 'Chaetodon_argentatus', 'Chaetodon_auriga', 'Chaetodon_auripes', 'Chaetodon_austriacus', 'Chaetodon_baronessa', 'Chaetodon_bennetti', 'Chaetodon_capistratus', 'Chaetodon_citrinellus', 'Chaetodon_decussatus', 'Chaetodon_ephippium', 'Chaetodon_falcula', 'Chaetodon_fasciatus', 'Chaetodon_flavirostris', 'Chaetodon_fremblii', 'Chaetodon_guentheri', 'Chaetodon_humeralis', 'Chaetodon_kleinii', 'Chaetodon_larvatus', 'Chaetodon_lineolatus', 'Chaetodon_lunula', 'Chaetodon_lunulatus', 'Chaetodon_melannotus', 'Chaetodon_melapterus', 'Chaetodon_mertensii', 'Chaetodon_meyeri', 'Chaetodon_miliaris', 'Chaetodon_modestus', 'Chaetodon_multicinctus', 'Chaetodon_ocellatus', 'Chaetodon_ocellicaudus', 'Chaetodon_octofasciatus', 'Chaetodon_ornatissimus', 'Chaetodon_oxycephalus', 'Chaetodon_paucifasciatus', 'Chaetodon_pelewensis', 'Chaetodon_plebeius', 'Chaetodon_punctatofasciatus', 'Chaetodon_quadrimaculatus', 'Chaetodon_rafflesii', 'Chaetodon_rainfordi', 'Chaetodon_reticulatus', 'Chaetodon_sedentarius', 'Chaetodon_semeion', 'Chaetodon_semilarvatus', 'Chaetodon_speculum', 'Chaetodon_striatus', 'Chaetodon_tinkeri', 'Chaetodon_triangulum', 'Chaetodon_trifascialis', 'Chaetodon_ulietensis', 'Chaetodon_unimaculatus', 'Chaetodon_vagabundus', 'Chaetodon_wiebeli', 'Chaetodon_xanthocephalus', 'Chaetodon_xanthurus', 'Chaetodontoplus_caeruleopunctatus', 'Chaetodontoplus_conspicillatus', 'Chaetodontoplus_duboulayi', 'Chaetodontoplus_melanosoma', 'Chaetodontoplus_meredithi', 'Chaetodontoplus_mesoleucus', 'Chaetodontoplus_septentrionalis', 'Chalinochromis_brichardi', 'Champsochromis_caeruleus', 'Channa_argus', 'Channa_asiatica', 'Channa_micropeltes', 'Channa_pleurophthalma', 'Chanos_chanos', 'Cheilinus_chlorourus', 'Cheilinus_fasciatus', 'Cheilinus_lunulatus', 'Cheilinus_trilobatus', 'Cheilinus_undulatus', 'Cheilio_inermis', 'Cheilochromis_euchilus', 'Cheilodactylus_nigripes', 'Cheilodactylus_vestitus', 'Cheilodipterus_artus', 'Cheilodipterus_intermedius', 'Cheilodipterus_macrodon', 'Cheilodipterus_quinquelineatus', 'Cheilopogon_abei', 'Chelidonichthys_kumu', 'Chelmon_marginalis', 'Chelmon_rostratus', 'Chelmonops_curiosus', 'Chelmonops_truncatus', 'Chelonodon_patoca', 'Chilatherina_bleheri', 'Chilodus_punctatus', 'Chilomycterus_antillarum', 'Chilomycterus_reticulatus', 'Chilomycterus_schoepfii', 'Chirocentrus_dorab', 'Chitala_blanci', 'Chitala_chitala', 'Chlorurus_bleekeri', 'Chlorurus_capistratoides', 'Chlorurus_enneacanthus', 'Chlorurus_frontalis', 'Chlorurus_gibbus', 'Chlorurus_perspicillatus', 'Chlorurus_sordidus', 'Choerodon_anchorago', 'Choerodon_fasciatus', 'Choerodon_schoenleinii', 'Chromis_agilis', 'Chromis_amboinensis', 'Chromis_analis', 'Chromis_atripectoralis', 'Chromis_caudalis', 'Chromis_cyanea', 'Chromis_insolata', 'Chromis_iomelas', 'Chromis_limbata', 'Chromis_margaritifer', 'Chromis_multilineata', 'Chromis_retrofasciata', 'Chromis_ternatensis', 'Chromis_viridis', 'Chromis_weberi', 'Chromis_xanthura', 'Chromobotia_macracanthus', 'Chrysiptera_biocellata', 'Chrysiptera_brownriggii', 'Chrysiptera_cyanea', 'Chrysiptera_hemicyanea', 'Chrysiptera_parasema', 'Chrysiptera_rollandi', 'Chrysiptera_starcki', 'Chrysiptera_talboti', 'Cichla_monoculus', 'Cirrhilabrus_cyanopleura', 'Cirrhilabrus_exquisitus', 'Cirrhilabrus_filamentosus', 'Cirrhilabrus_jordani', 'Cirrhilabrus_laboutei', 'Cirrhilabrus_lineatus', 'Cirrhilabrus_scottorum', 'Cirrhilabrus_temminckii', 'Cirrhitichthys_aprinus', 'Cirrhitichthys_aureus', 'Cirrhitichthys_falco', 'Cirrhitichthys_oxycephalus', 'Cirrhitops_fasciatus', 'Cirrhitus_pinnulatus', 'Cleidopus_gloriamaris', 'Cleithracara_maronii', 'Clepticus_parrae', 'Clupea_harengus', 'Cnidoglanis_macrocephalus', 'Colomesus_psittacus', 'Colossoma_macropomum', 'Cookeolus_japonicus', 'Copadichromis_azureus', 'Copadichromis_borleyi', 'Copadichromis_pleurostigma', 'Copella_arnoldi', 'Coradion_altivelis', 'Coradion_chrysozonus', 'Coris_aygula', 'Coris_batuensis', 'Coris_caudimacula', 'Coris_picta', 'Corythoichthys_haematopterus', 'Cryptocentrus_cinctus', 'Cryptocentrus_fasciatus', 'Cryptocentrus_leptocephalus', 'Ctenochaetus_binotatus', 'Ctenochaetus_tominiensis', 'Ctenochaetus_truncatus', 'Ctenogobiops_feroculus', 'Ctenogobiops_tangaroai', 'Cyclichthys_orbicularis', 'Cyclichthys_spilostylus', 'Cypho_purpurascens', 'Cyprinocirrhites_polyactis', 'Cyprinus_carpio', 'Cypselurus_poecilopterus', 'Dactylopterus_volitans', 'Dascyllus_aruanus', 'Dascyllus_melanurus', 'Dascyllus_reticulatus', 'Dascyllus_trimaculatus', 'Dasyatis_americana', 'Decapterus_macarellus', 'Dendrochirus_biocellatus', 'Dendrochirus_brachypterus', 'Dendrochirus_zebra', 'Diademichthys_lineatus', 'Diagramma_pictum', 'Diodon_holocanthus', 'Diodon_hystrix', 'Diodon_liturosus', 'Diplectrum_formosum', 'Diploprion_bifasciatum', 'Diproctacanthus_xanthurus', 'Dischistodus_perspicillatus', 'Dischistodus_prosopotaenia', 'Dunckerocampus_pessuliferus', 'Echeneis_naucrates', 'Echidna_nebulosa', 'Ecsenius_bicolor', 'Ecsenius_lineatus', 'Ecsenius_midas', 'Elacatinus_oceanops', 'Eleutheronema_tetradactylum', 'Emblemaria_pandionis', 'Enchelycore_pardalis', 'Enoplosus_armatus', 'Epibulus_insidiator', 'Epinephelus_adscensionis', 'Epinephelus_aeneus', 'Epinephelus_akaara', 'Epinephelus_amblycephalus', 'Epinephelus_areolatus', 'Epinephelus_bleekeri', 'Epinephelus_coeruleopunctatus', 'Epinephelus_coioides', 'Epinephelus_costae', 'Epinephelus_cyanopodus', 'Epinephelus_epistictus', 'Epinephelus_fasciatus', 'Epinephelus_flavocaeruleus', 'Epinephelus_fuscoguttatus', 'Epinephelus_guttatus', 'Epinephelus_itajara', 'Epinephelus_labriformis', 'Epinephelus_maculatus', 'Epinephelus_malabaricus', 'Epinephelus_marginatus', 'Epinephelus_merra', 'Epinephelus_morio', 'Epinephelus_spilotoceps', 'Epinephelus_striatus', 'Epinephelus_summana', 'Epinephelus_tauvina', 'Epinephelus_tukula', 'Equetus_punctatus', 'Esox_lucius', 'Etelis_carbunculus', 'Etelis_coruscans', 'Eurypegasus_draconis', 'Eviota_guttata', 'Evistias_acutirostris', 'Exallias_brevis', 'Fistularia_commersonii', 'Fistularia_tabacaria', 'Forcipiger_flavissimus', 'Forcipiger_longirostris', 'Fusigobius_inframaculatus', 'Gadus_morhua', 'Galeocerdo_cuvier', 'Gambusia_holbrooki', 'Genicanthus_bellus', 'Genicanthus_lamarck', 'Genicanthus_melanospilos', 'Genicanthus_watanabei', 'Gerres_filamentosus', 'Ginglymostoma_cirratum', 'Girella_zebra', 'Gnathanodon_speciosus', 'Gnathodentex_aureolineatus', 'Gnatholepis_anjerensis', 'Gobiodon_citrinus', 'Gobiodon_histrio', 'Gobiodon_okinawae', 'Gomphosus_varius', 'Gorgasia_preclara', 'Gramma_loreto', 'Gymnomuraena_zebra', 'Gymnosarda_unicolor', 'Gymnothorax_breedeni', 'Gymnothorax_favagineus', 'Gymnothorax_fimbriatus', 'Gymnothorax_flavimarginatus', 'Gymnothorax_funebris', 'Gymnothorax_javanicus', 'Gymnothorax_meleagris', 'Gymnothorax_miliaris', 'Gymnothorax_moringa', 'Gymnothorax_nudivomer', 'Gymnothorax_prasinus', 'Gymnothorax_thyrsoideus', 'Gymnothorax_undulatus', 'Haemulon_aurolineatum', 'Haemulon_carbonarium', 'Haemulon_chrysargyreum', 'Haemulon_flavolineatum', 'Haemulon_sciurus', 'Halicampus_macrorhynchus', 'Halichoeres_argus', 'Halichoeres_biocellatus', 'Halichoeres_bivittatus', 'Halichoeres_chloropterus', 'Halichoeres_chrysus', 'Halichoeres_garnoti', 'Halichoeres_hartzfeldii', 'Halichoeres_hortulanus', 'Halichoeres_maculipinna', 'Halichoeres_marginatus', 'Halichoeres_melanurus', 'Halichoeres_nebulosus', 'Halichoeres_prosopeion', 'Halichoeres_radiatus', 'Halichoeres_scapularis', 'Halichoeres_trimaculatus', 'Halophryne_diemensis', 'Hemigymnus_fasciatus', 'Hemigymnus_melapterus', 'Hemitaurichthys_polylepis', 'Heniochus_acuminatus', 'Heniochus_chrysostomus', 'Heniochus_diphreutes', 'Heniochus_monoceros', 'Heniochus_singularius', 'Heniochus_varius', 'Heteroconger_hassi', 'Heterodontus_galeatus', 'Heterodontus_portusjacksoni', 'Heterodontus_zebra', 'Heteropriacanthus_cruentatus', 'Hippocampus_erectus', 'Hippocampus_reidi', 'Histrio_histrio', 'Holacanthus_bermudensis', 'Holacanthus_ciliaris', 'Holacanthus_tricolor', 'Holocentrus_rufus', 'Hologymnosus_doliatus', 'Hoplolatilus_starcki', 'Hypoplectrodes_maccullochi', 'Hypoplectrus_indigo', 'Hypoplectrus_puella', 'Hypoplectrus_unicolor', 'Iniistius_aneitensis', 'Iniistius_pavo', 'Inimicus_didactylus', 'Iriatherina_werneri', 'Istigobius_decoratus', 'Isurus_oxyrinchus', 'Kajikia_audax', 'Katsuwonus_pelamis', 'Kuhlia_mugil', 'Kuhlia_rupestris', 'Kyphosus_bigibbus', 'Kyphosus_cinerascens', 'Kyphosus_sectatrix', 'Kyphosus_vaigiensis', 'Labracinus_cyclophthalmus', 'Labrichthys_unilineatus', 'Labrisomus_nuchipinnis', 'Labroides_dimidiatus', 'Labropsis_australis', 'Labrus_bergylta', 'Lachnolaimus_maximus', 'Lactophrys_bicaudalis', 'Lactophrys_trigonus', 'Lactophrys_triqueter', 'Lactoria_cornuta', 'Lactoria_fornasini', 'Lagocephalus_sceleratus', 'Lampris_guttatus', 'Lethrinus_erythracanthus', 'Lethrinus_harak', 'Lethrinus_microdon', 'Lethrinus_nebulosus', 'Lobotes_surinamensis', 'Lubricogobius_exiguus', 'Lutjanus_analis', 'Lutjanus_apodus', 'Lutjanus_argentiventris', 'Lutjanus_bengalensis', 'Lutjanus_biguttatus', 'Lutjanus_bohar', 'Lutjanus_campechanus', 'Lutjanus_carponotatus', 'Lutjanus_cyanopterus', 'Lutjanus_decussatus', 'Lutjanus_ehrenbergii', 'Lutjanus_fulviflamma', 'Lutjanus_fulvus', 'Lutjanus_gibbus', 'Lutjanus_griseus', 'Lutjanus_jocu', 'Lutjanus_kasmira', 'Lutjanus_mahogoni', 'Lutjanus_quinquelineatus', 'Lutjanus_sebae', 'Lutjanus_synagris', 'Macolor_macularis', 'Macolor_niger', 'Macropharyngodon_choati', 'Macropharyngodon_meleagris', 'Macropharyngodon_negrosensis', 'Macropharyngodon_ornatus', 'Malacanthus_brevirostris', 'Malacanthus_latovittatus', 'Malacanthus_plumieri', 'Malacoctenus_triangulatus', 'Manta_birostris', 'Masturus_lanceolatus', 'Megalops_atlanticus', 'Meiacanthus_atrodorsalis', 'Melanotaenia_australis', 'Melichthys_niger', 'Melichthys_vidua', 'Mene_maculata', 'Meuschenia_hippocrepis', 'Microcanthus_strigatus', 'Microspathodon_chrysurus', 'Mogurnda_adspersa', 'Mola_mola', 'Monocentris_japonica', 'Monodactylus_argenteus', 'Monotaxis_grandoculis', 'Mugil_cephalus', 'Mulloidichthys_flavolineatus', 'Mulloidichthys_martinicus', 'Mulloidichthys_vanicolensis', 'Mycteroperca_bonaci', 'Mycteroperca_tigris', 'Myrichthys_breviceps', 'Myripristis_adusta', 'Myripristis_berndti', 'Myripristis_jacobus', 'Myripristis_kuntee', 'Myripristis_murdjan', 'Myripristis_pralinia', 'Myripristis_violacea', 'Myripristis_vittata', 'Naso_annulatus', 'Naso_brachycentron', 'Naso_brevirostris', 'Naso_elegans', 'Naso_hexacanthus', 'Naso_lituratus', 'Naso_unicornis', 'Naucrates_ductor', 'Negaprion_brevirostris', 'Nemateleotris_decora', 'Nemateleotris_magnifica', 'Neoceratodus_forsteri', 'Neocirrhites_armatus', 'Neoglyphidodon_melas', 'Neoglyphidodon_nigroris', 'Neoglyphidodon_oxyodon', 'Neoniphon_sammara', 'Neosilurus_ater', 'Notolabrus_tetricus', 'Novaculichthys_taeniourus', 'Novaculoides_macrolepidotus', 'Odonus_niger', 'Ogcocephalus_cubifrons', 'Ophichthus_bonaparti', 'Ophthalmolepis_lineolata', 'Orectolobus_halei', 'Orectolobus_maculatus', 'Ostorhinchus_aureus', 'Ostorhinchus_cyanosoma', 'Ostracion_cubicus', 'Ostracion_meleagris', 'Ostracion_solorensis', 'Oxycheilinus_bimaculatus', 'Oxycheilinus_celebicus', 'Oxycheilinus_digramma', 'Oxycheilinus_unifasciatus', 'Oxycirrhites_typus', 'Oxymonacanthus_longirostris', 'Parablennius_marmoreus', 'Paracaesio_xanthura', 'Paracanthurus_hepatus', 'Paracentropyge_multifasciata', 'Parachaetodon_ocellatus', 'Paracheilinus_filamentosus', 'Paracheilinus_flavianalis', 'Paracirrhites_arcatus', 'Paracirrhites_forsteri', 'Paracirrhites_hemistictus', 'Paraluteres_prionurus', 'Parapercis_clathrata', 'Parapercis_millepunctata', 'Parapercis_schauinslandii', 'Parapercis_snyderi', 'Parapercis_xanthozona', 'Paraplesiops_bleekeri', 'Paraplesiops_meleagris', 'Pareques_acuminatus', 'Parma_microlepis', 'Parupeneus_barberinoides', 'Parupeneus_barberinus', 'Parupeneus_crassilabris', 'Parupeneus_cyclostomus', 'Parupeneus_heptacanthus', 'Parupeneus_macronemus', 'Parupeneus_multifasciatus', 'Parupeneus_spilurus', 'Parupeneus_trifasciatus', 'Pataecus_fronto', 'Pelates_quadrilineatus', 'Pempheris_oualensis', 'Pentaceropsis_recurvirostris', 'Perca_fluviatilis', 'Phycodurus_eques', 'Phyllopteryx_taeniolatus', 'Pictichromis_paccagnellae', 'Platax_batavianus', 'Platax_orbicularis', 'Platax_pinnatus', 'Platax_teira', 'Plectorhinchus_albovittatus', 'Plectorhinchus_chaetodonoides', 'Plectorhinchus_flavomaculatus', 'Plectorhinchus_lineatus', 'Plectorhinchus_picus', 'Plectorhinchus_polytaenia', 'Plectorhinchus_vittatus', 'Plectroglyphidodon_dickii', 'Plectroglyphidodon_lacrymatus', 'Pleurosicya_micheli', 'Plotosus_lineatus', 'Pomacanthus_annularis', 'Pomacanthus_arcuatus', 'Pomacanthus_imperator', 'Pomacanthus_navarchus', 'Pomacanthus_paru', 'Pomacanthus_semicirculatus', 'Pomacanthus_sexstriatus', 'Pomacanthus_xanthometopon', 'Pomacentrus_alleni', 'Pomacentrus_amboinensis', 'Pomacentrus_auriventris', 'Pomacentrus_bankanensis', 'Pomacentrus_brachialis', 'Pomacentrus_coelestis', 'Pomacentrus_moluccensis', 'Pomacentrus_pavo', 'Pomacentrus_reidi', 'Pomacentrus_vaiuli', 'Premnas_biaculeatus', 'Priacanthus_blochii', 'Priacanthus_hamrur', 'Priolepis_cincta', 'Prionace_glauca', 'Prionotus_ophryas', 'Pristiapogon_kallopterus', 'Pseudalutarius_nasicornis', 'Pseudanthias_bicolor', 'Pseudanthias_cooperi', 'Pseudanthias_dispar', 'Pseudanthias_fasciatus', 'Pseudanthias_huchtii', 'Pseudanthias_hypselosoma', 'Pseudanthias_lori', 'Pseudanthias_pascalus', 'Pseudanthias_pleurotaenia', 'Pseudanthias_rubrizonatus', 'Pseudanthias_smithvanizi', 'Pseudanthias_squamipinnis', 'Pseudanthias_tuka', 'Pseudanthias_ventralis', 'Pseudobalistes_flavimarginatus', 'Pseudobalistes_fuscus', 'Pseudocheilinus_evanidus', 'Pseudocheilinus_hexataenia', 'Pseudocheilinus_ocellatus', 'Pseudocheilinus_octotaenia', 'Pseudochromis_fuscus', 'Pseudodax_moluccanus', 'Pseudojuloides_cerasinus', 'Pseudupeneus_maculatus', 'Pteragogus_cryptus', 'Pteragogus_enneacanthus', 'Ptereleotris_evides', 'Ptereleotris_zebra', 'Pterocaesio_tile', 'Pterois_antennata', 'Pterois_lunulata', 'Pterois_miles', 'Pterois_radiata', 'Pterois_volitans', 'Pterosynchiropus_splendidus', 'Puntius_conchonius', 'Pygoplites_diacanthus', 'Rachycentron_canadum', 'Ranzania_laevis', 'Rastrelliger_kanagurta', 'Rhina_ancylostoma', 'Rhincodon_typus', 'Rhinecanthus_aculeatus', 'Rhinecanthus_rectangulus', 'Rhinecanthus_verrucosus', 'Rhinomuraena_quaesita', 'Rhinopias_eschmeyeri', 'Rhinopias_frondosa', 'Rhinoptera_bonasus', 'Rhizoprionodon_terraenovae', 'Rhynchobatus_australiae', 'Rhynchostracion_nasus', 'Richardsonichthys_leucogaster', 'Rocio_octofasciata', 'Rypticus_saponaceus', 'Salarias_ramosus', 'Sargocentron_caudimaculatum', 'Sargocentron_cornutum', 'Sargocentron_diadema', 'Sargocentron_spiniferum', 'Scarus_altipinnis', 'Scarus_chameleon', 'Scarus_coeruleus', 'Scarus_flavipectoralis', 'Scarus_forsteni', 'Scarus_frenatus', 'Scarus_ghobban', 'Scarus_guacamaia', 'Scarus_iseri', 'Scarus_niger', 'Scarus_oviceps', 'Scarus_psittacus', 'Scarus_quoyi', 'Scarus_rubroviolaceus', 'Scarus_schlegeli', 'Scarus_taeniopterus', 'Scarus_tricolor', 'Scarus_vetula', 'Scatophagus_argus', 'Scleropages_jardinii', 'Scolopsis_affinis', 'Scolopsis_bilineata', 'Scolopsis_vosmeri', 'Scomberoides_commersonnianus', 'Scorpaena_brasiliensis', 'Scorpaena_plumieri', 'Scorpis_aequipinnis', 'Sebastapistes_cyanostigma', 'Selenotoca_multifasciata', 'Seriola_rivoliana', 'Serranocirrhitus_latus', 'Serranus_baldwini', 'Serranus_tigrinus', 'Sicyopterus_lagocephalus', 'Siganus_doliatus', 'Siganus_punctatus', 'Siganus_stellatus', 'Siganus_unimaculatus', 'Siganus_virgatus', 'Siganus_vulpinus', 'Signigobius_biocellatus', 'Solenostomus_halimeda', 'Sparisoma_aurofrenatum', 'Sparisoma_chrysopterum', 'Sparisoma_radians', 'Sparisoma_rubripinne', 'Sparisoma_viride', 'Sphaeramia_nematoptera', 'Sphoeroides_spengleri', 'Sphyraena_barracuda', 'Sphyraena_qenie', 'Sphyrna_lewini', 'Sphyrna_mokarran', 'Sphyrna_tiburo', 'Stegastes_fasciolatus', 'Stegastes_leucostictus', 'Stegastes_nigricans', 'Stegastes_partitus', 'Stegastes_planifrons', 'Stegastes_variabilis', 'Stegostoma_fasciatum', 'Stephanolepis_hispidus', 'Stethojulis_bandanensis', 'Stonogobiops_xanthorhinica', 'Sufflamen_bursa', 'Sufflamen_chrysopterum', 'Symphorichthys_spilurus', 'Synanceia_verrucosa', 'Synodus_intermedius', 'Synodus_variegatus', 'Taeniamia_fucata', 'Taenianotus_triacanthus', 'Taeniura_lymma', 'Taeniurops_meyeni', 'Tetractenos_glaber', 'Thalassoma_amblycephalum', 'Thalassoma_bifasciatum', 'Thalassoma_hardwicke', 'Thalassoma_jansenii', 'Thalassoma_lucasanum', 'Thalassoma_lunare', 'Thalassoma_lutescens', 'Thalassoma_purpureum', 'Thalassoma_quinquevittatum', 'Thalassoma_trilobatum', 'Thunnus_albacares', 'Tilodon_sexfasciatus', 'Tinca_tinca', 'Toxotes_chatareus', 'Toxotes_jaculatrix', 'Trachichthys_australis', 'Trachinotus_falcatus', 'Triaenodon_obesus', 'Trimma_benjamini', 'Trygonoptera_testacea', 'Trygonorrhina_fasciata', 'Tylosurus_crocodilus', 'Upeneichthys_lineatus', 'Upeneus_tragula', 'Urobatis_jamaicensis', 'Valenciennea_helsdingenii', 'Valenciennea_puellaris', 'Valenciennea_strigata', 'Valenciennea_wardii', 'Variola_albimarginata', 'Variola_louti', 'Wetmorella_nigropinnata', 'Xanthichthys_auromarginatus', 'Xanthichthys_caeruleolineatus', 'Xiphophorus_hellerii', 'Xyrichtys_splendens', 'Zanclus_cornutus', 'Zebrasoma_desjardinii', 'Zebrasoma_scopas']

print('Loading Weights ...')
model_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))

print('Adding Average Pooling Layer and Softmax Output Layer ...')
# Freeze the model except the last layer
model_notop.trainable = False
#model_notop.layers[-1].trainable = True

output = model_notop.get_layer(index = -1).output
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(985, activation='softmax', name='predictions')(output)

model = Model(model_notop.input, output)
model.summary()

optimizer = SGD(momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer = optimizer,  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)],)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

# this is the augmentation configuration we will use for validation:
val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        #nsw_train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
es01 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
#
# # # Set class weights for imbalanced data
# # class_weights = class_weight.compute_class_weight('balanced',
# #                                                  np.unique(train_generator.classes),
# #                                                  train_generator.classes)
#
#
history = model.fit(
        train_generator,
        steps_per_epoch = nbr_train_samples/batch_size,
        epochs = nbr_epochs,
        validation_data = validation_generator,
        validation_steps = nbr_validation_samples/batch_size,
        callbacks=[es01])

# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
model_notop.trainable = True
model.summary()

optimizer = SGD(learning_rate = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer = optimizer,  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)],)
#
# # autosave best Model
best_model_file = 'InceptionV3_weights01.h5'
callback01 = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)

# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
#es01 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
        train_generator,
        steps_per_epoch = nbr_train_samples/batch_size,
        epochs = 25,
        validation_data = validation_generator,
        validation_steps = nbr_validation_samples/batch_size,
        callbacks=[callback01])

# Train the top layer on the NSW fish dataset
nbr_nsw_train_samples = 1421
nbr_nsw_validation_samples = 483
nbr_nsw_epochs = 50

nsw_train_dir = 'C:\\Users\\bxiong\\Downloads\\NSW\\NSWFish_train'
nsw_validation_dir = 'C:\\Users\\bxiong\\Downloads\\NSW\\NSWFish_validation'
nsw_test_dir = 'C:\\Users\\bxiong\\Downloads\\NSW\\NSWFish_test'

FishNames01 = ['Albacore Tuna', 'Amberjack', 'Atlantic Salmon', 'Australian Salmon', 'Australian Bass', 'Australian Bonito', 'Australian Sawtail', 'Banded Morwong', 'Banded Rockcod', 'Bass Groper', 'Big-Eye Tuna', 'Black Marlin', 'Blue Drummer', 'Blue Marlin', 'Blue-Eye Trevalla', 'Bream', 'Brook Trout', 'Brown Trout', 'Cobia', 'Eastern (Freshwater) Cod', 'Eastern Red Scorpionfish', 'Estuary Perch', 'Flathead Dusky', 'Flounder', 'Freshwater Catfish (Eel-tailed)', 'Garfish Eastern Sea', 'Gemfish', 'Golden Perch', 'Grey Morwong', 'Groper BlueRedBrown', 'Hairtail', 'Hapuku', 'Jackass Morwong', 'Leatherjacket', 'Longfin Eel', 'Longtail Tuna', 'Luderick', 'Macquarie Perch', 'Mahi Mahi', 'Mangrove Jack', 'Moses Snapper (Perch)', 'Mullet Poddy All others', 'Mulloway (Jewfish)', 'Murray Cod', 'Pearl Perch', 'Rainbow Trout', 'Red Morwong', 'Rock Blackfish (Black Drummer)', 'Sailfish', 'Samsonfish', 'Shortfin Eel', 'Silver Perch', 'Snapper', 'Southern Bluefin Tuna', 'Spanish Mackerel', 'Spearfish', 'Spotted Mackerel', 'Striped Marlin', 'Swordfish', 'Tailor', 'Tarwhine', 'Teraglin', 'Trevally', 'Trout Cod', 'Wahoo', 'Whiting', 'Yellowfin Tuna', 'Yellowtail Kingfish']

# Remove the last several layers
model2= Model(inputs=model.input, outputs=model.layers[-4].output)

print('Adding Average Pooling Layer and Softmax Output Layer ...')
# Freeze the model except the last layer
model2.trainable = False

output = model2.get_layer(index = -1).output
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(68, activation='softmax', name='predictions')(output)

new_model = Model(model2.input, output)
new_model.summary()

optimizer = SGD(momentum = 0.9, decay = 0.0, nesterov = True)
new_model.compile(loss='categorical_crossentropy', optimizer = optimizer,  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)],)

nsw_train_generator = train_datagen.flow_from_directory(
        nsw_train_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = FishNames01,
        class_mode = 'categorical')

nsw_validation_generator = val_datagen.flow_from_directory(
        nsw_validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        #save_prefix = 'aug',
        classes = FishNames01,
        class_mode = 'categorical')

# This callback will stop the training when there is no improvement in
# the loss for three consecutive epochs.
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# # Set class weights for imbalanced data
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(nsw_train_generator.classes),
#                                                  nsw_train_generator.classes)

history01 = new_model.fit(
        nsw_train_generator,
        steps_per_epoch = nbr_nsw_train_samples/batch_size,
        epochs = nbr_epochs,
        validation_data = nsw_validation_generator,
        validation_steps = nbr_nsw_validation_samples/batch_size,
        callbacks = [es])

# # plot training history
pyplot.plot(history01.history['loss'], label='train')
pyplot.plot(history01.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
model2.trainable = True

new_model.summary()

optimizer = SGD(learning_rate = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
new_model.compile(loss='categorical_crossentropy', optimizer = optimizer,  metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=5)],)


# early stopping and autosave best Model
best_model_file = 'InceptionV3_weights02.h5'
best_model02 = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True)
#es02 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history02 = new_model.fit(
        nsw_train_generator,
        steps_per_epoch = nbr_nsw_train_samples/batch_size,
        epochs = 25,
        validation_data = nsw_validation_generator,
        validation_steps = nbr_nsw_validation_samples/batch_size,
        callbacks = [best_model02])

# plot training history
pyplot.plot(history02.history['loss'], label='train')
pyplot.plot(history02.history['val_loss'], label='validation')
pyplot.legend()
pyplot.show()

# Predict on test set

test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
)

test_generator = test_datagen.flow_from_directory(
        nsw_test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

# print('Loading model and weights from training process ...')

final_model = models.load_model("./InceptionV3_weights02.h5")

print('Begin to predict for testing data ...')
score = top_k_accuracy_score(test_generator.classes, final_model.predict(test_generator), k=5, labels=np.arange(68))
print('Test accuracy:', score)

