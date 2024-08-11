pi_value = math.pi
def generate_SS_map():
    num = 0
    for i in range(0, 2):
        print('###################################################################')
        print(f"start generating for i = {i}, and 41*i = {41*i}, y = {-644+(41*i)}")
        for j in range(0, 990, 33):
            print(f"j = {j}")
            for k in range(4):
                print(f"orientation {k}")
                n = randint(1, 12)
                scene_building = load_scene('building_xml/building29.xml')
                # scene_plane = load_scene('Plane/Plane1.xml')
            
                ######### scene_building settings ###############################
                scene_building.tx_array = PlanarArray(num_rows=1,
                                                  num_cols=1,
                                                  vertical_spacing=0.5,
                                                  horizontal_spacing=0.5,
                                                  pattern="tr38901",
                                                  polarization="VH")

                # Configure antenna array for all receivers
                scene_building.rx_array = PlanarArray(num_rows=1,
                                                   num_cols=1,
                                                   vertical_spacing=0.5,
                                                   horizontal_spacing=0.5,
                                                   pattern="iso",
                                                   polarization="V")

                # Create transmitter
                tx_b = Transmitter(name="tx_b",
                          position=[-520+j,-644+(41*i), 24],
                          orientation=[(2*pi_value)/n,0,0],
                          color=(1, 0, 0))
                
                scene_building.add(tx_b)
                scene_building.frequency = 3.66e9 # in Hz; implicitly updates RadioMaterials
                scene_building.synthetic_array = True
                cm = scene_building.coverage_map(max_depth=8, 
                                            los=True, 
                                            reflection=True, 
                                            diffraction=True, 
                                            check_scene=False)
                ##################################################################
            
                # change coverage map into tensor
                cm_tensor = cm.as_tensor()
                cm_2D = cm_tensor.numpy()[0, :, :]
                cm_2D = np.flip(cm_2D[::-1])
                
                # cm_2D = np.resize(cm_2D, (128, 128))
            
                # change W into dB
                cm_db = 10 * np.log10(cm_2D)
                shape = cm_db.shape
                # print(shape)
            
            
                ###################### Generating SS map #######################################################
                # Transmitter antenna power heatmap
                p = np.random.uniform(10, 36)
                P_Tx = np.full(shape, p)
            
                #Transmitter antenna gain heatmap
                gt = np.random.uniform(10, 21)
                G_Tx = np.full(shape, gt)

                # receiver antenna gain heat map
                gr = np.random.uniform(10, 21)
                G_Rx = np.full(shape, gr)
            
                # Potential insertion loss of the link
                il = np.random.uniform(-10, 11)
                IL = np.full(shape, il)
            
                # calculate SS map
                S = P_Tx + G_Tx + cm_db + G_Rx - IL
            
                np.save(f'auto_generated_data/SSmap/SSmap29_{num}_{k}.npy', S)
                print(f"SSmap29_{num}_{k}.npy saved")
                #################################################################################################
            
                ############################## generating sparse SS map #########################################
                Sparse_SS = np.zeros(shape = shape)

                num_points = randint(1, 201)
                
                rows = np.random.randint(0, S.shape[0], num_points)
                cols = np.random.randint(0, S.shape[1], num_points)
    
                for row, col in zip(rows, cols):
                    Sparse_SS[row, col] = S[row, col]  

                np.save(f'auto_generated_data/Sparse_SSmap/Sparse_SSmap29_{num}_{k}.npy', Sparse_SS)
                print(f"Sparse_SSmap29_{num}_{k}.npy saved")
                #################################################################################################
            num += 1
            
generate_SS_map()
