# Written by Muhammad Sarmad
# Date : 23 August 2018



from RL_params import *
from Datasets.plyfile.plyfile import PlyData
from visu_util import plot_pcd_three_views

np.random.seed(5)
#torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)


def main(args,vis_Valid,vis_Valida):
    """ Transforms/ Data Augmentation Tec """
    co_transforms = pc_transforms.Compose([])

    input_transforms = transforms.Compose([

        pc_transforms.ArrayToTensor(),
        #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
        #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])

    """-----------------------------------------------Data Loader----------------------------------------------------"""

    if (args.net_name == 'auto_encoder'):

        [test_dataset,_] = Datasets.__dict__[args.dataName](input_root=args.data_incomplete_test,
                                                                          target_root=None,
                                                                          split=1.0,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms,
                                                                          give_name = True)




    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)


    """----------------Model Settings-----------------------------------------------"""

    print('Encoder Model: {0}, Decoder Model : {1}'.format(args.model_encoder,args.model_decoder))
    print('GAN Model Generator:{0} & Discriminator : {1} '.format(args.model_generator,args.model_discriminator))


    network_data_AE = torch.load(args.pretrained_enc_dec)

    network_data_G = torch.load(args.pretrained_G)

    network_data_D = torch.load(args.pretrained_D)

    network_data_Actor = torch.load(args.pretrained_Actor)

    network_data_Critic = torch.load(args.pretrained_Critic)

    model_encoder = models.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                        data=network_data_AE, calc_loss=False).cuda()
    model_decoder = models.__dict__[args.model_decoder](args, data=network_data_AE).cuda()

    model_G = models.__dict__[args.model_generator](args, data=network_data_G).cuda()

    model_D = models.__dict__[args.model_discriminator](args, data=network_data_D).cuda()

    model_actor = models.__dict__['actor_net'](args, data=network_data_Actor).cuda()

    model_critic = models.__dict__['critic_net'](args, data=network_data_Critic).cuda()



    params = get_n_params(model_encoder)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(model_decoder)
    print('| Number of Decoder parameters [' + str(params) + ']...')

    params = get_n_params(model_actor)
    print('| Number of Actor parameters [' + str(params) + ']...')

    params = get_n_params(model_critic)
    print('| Number of Critic parameters [' + str(params) + ']...')


    chamfer = ChamferLoss(args)
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)

    epoch = 0


    test_loss = testRL(test_loader, model_encoder, model_decoder, model_G,model_D, model_actor,model_critic,epoch, args, chamfer,nll, mse, norm, vis_Valid,
                     vis_Valida)
    print('Average Loss :{}'.format(test_loss))


def testRL(test_loader,model_encoder,model_decoder, model_G,model_D,model_actor,model_critic,epoch,args, chamfer,nll, mse,norm,vis_Valid,vis_Valida):

    model_encoder.eval()
    model_decoder.eval()
    model_G.eval()
    model_D.eval()
    model_actor.eval()
    model_critic.eval()

    num_test_episodes = len(test_loader)

    avg_reward = 0.
    env = envs(args, model_G, model_D, model_encoder, model_decoder, num_test_episodes)
    epi_timestep_list = []

    for i, (input,fname) in enumerate(test_loader):

        obs = env.agent_input(input)  # env(input, action_rand)
        done = False
        step = 0
        while not done:
            # Action By Agent and collect reward
            if args.policy_name=="DDPG":
                action = model_actor(np.array(obs))
            elif args.policy_name=="SoftAC":
                action = model_actor(np.array(obs), deterministic=True)

            action = torch.tensor(action).cuda().unsqueeze(dim=0)
            new_obs, new_pc, reward, done, _ = env(input,
                                                     action,
                                                     render=True,
                                                     disp=True,
                                                     is_first=True if step==0 else False,
                                                     fname=fname,
                                                     filenum = str(i),
                                                     step=step)

            avg_reward += reward
            obs = new_obs
            input = new_pc

            done = True if done or step == args.max_episodes_steps + 1 else False
            step += 1

        epi_timestep_list.append(step)

    avg_reward /= num_test_episodes
    avg_cd_loss = env.cd_loss / num_test_episodes
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (num_test_episodes, avg_reward))
    print("Average episode_timestep: " + str(sum(epi_timestep_list) / len(epi_timestep_list)))
    print("Average CD loss: " + str(avg_cd_loss))
    print("---------------------------------------")
    with open(os.path.join('test', 'results'), 'w') as log:
        log.write("Evaluation over %d episodes: %f \n" %(num_test_episodes, avg_reward))
        log.write("Average episode_timestep: " + str(sum(epi_timestep_list) / len(epi_timestep_list)) + '\n')
        log.write("Average CD loss: " + str(avg_cd_loss) + '\n')


class envs(nn.Module):
    def __init__(self,args,model_G,model_D,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()

        self.nll = NLL()
        self.mse = MSE(reduction='elementwise_mean')
        self.norm = Norm(dims=args.z_dim)
        self.chamfer = ChamferLoss(args)
        self.epoch = 0
        self.epoch_size =epoch_size

        self.model_G = model_G
        self.model_D = model_D
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.figures = 3
        self.attempts = args.attempts
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.lossess = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
        self.i = 0
        self.prev_reward_D = -1000000
        self.cd_loss = 0
    def reset(self, epoch_size, is_training, figures =3):
        self.is_training = is_training
        self.j = 1;
        self.figures = figures;
        self.epoch_size= epoch_size
    def agent_input(self,input): # input: incomplete point cloud
        with torch.no_grad():
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            encoder_out = self.model_encoder(input_var, )
            out = encoder_out.detach().cpu().numpy().squeeze()
        return out
    def forward(self,input,action,render=False, disp=False, is_first=False, fname=None, filenum=None, step=0):
        with torch.no_grad():
            # Encoder Input
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)

            # Encoder  output
            encoder_out = self.model_encoder(input_var, )

            # D Decoder Output
#            pc_1, pc_2, pc_3 = self.model_decoder(encoder_out)
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = Variable(action, requires_grad=True).cuda()

            # Generator Output
            out_GD, _ = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            out_G = out_G.contiguous().view(-1, args.state_dim)

            # Discriminator Output
            #out_D, _ = self.model_D(encoder_out.view(-1,1,32,32))
        #    out_D, _ = self.model_D(encoder_out.view(-1, 1, 1,args.state_dim)) # TODO Alert major mistake
            out_D, _ = self.model_D(out_GD) # TODO Alert major mistake

            # H Decoder Output
#            pc_1_G, pc_2_G, pc_3_G = self.model_decoder(out_G)
            pc_1_G = self.model_decoder(out_G)


            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input_var, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)

            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC


        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(encoder_out, out_G)

        # Norm Loss
        #loss_norm = self.norm(z)

        # Chamfer loss
        #loss_chamfer = self.chamfer(pc_1_G, pc_1)  # #self.chamfer(pc_1_G, trans_input) instantaneous loss of batch items
        loss_chamfer = self.chamfer(pc_1_G, trans_input)

        # States Formulation
        state_curr = np.array([loss_D.cpu().data.numpy(), loss_GFV.cpu().data.numpy()
                                  , loss_chamfer.cpu().data.numpy()])
      #  state_prev = self.state_prev

        reward_D = state_curr[0]#state_curr[0] - self.state_prev[0]
        reward_GFV = -state_curr[1]# -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2]#-state_curr[2] + self.state_prev[2]
        #reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = reward_D * 0.01 + reward_GFV * 10.0 + reward_chamfer * 100.0 #+ reward_norm * 1/10

        #self.lossess.update(loss_chamfer.item(), input.size(0))  # loss and batch size as input

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

      #  if self.j <= 5:
        visuals = OrderedDict(
            [('Input_pc', trans_input_temp.detach().cpu().numpy()),
             ('AE Predicted_pc', pc_1_temp.detach().cpu().numpy()),
             ('GAN Generated_pc', pc_1_G_temp.detach().cpu().numpy())])
        if render==True and self.j <= self.figures:
         vis_Valida[self.j].display_current_results(visuals, self.epoch, self.i)
         self.j += 1

        if disp:
            #print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1

        test1 = trans_input_temp.detach().cpu().numpy()
        test2 = pc_1_temp.detach().cpu().numpy()
        test3 = pc_1_G_temp.detach().cpu().numpy()

        fname = fname[0]

        category_id, model_id = fname.split('/')
        if not os.path.exists(os.path.join('test', category_id)):
            os.makedirs(os.path.join('test', category_id))


        if not os.path.exists(os.path.join('test', category_id, model_id)):
            os.makedirs(os.path.join('test', category_id, model_id))

        np.savetxt(os.path.join('test', category_id, model_id, str(step) + '_input.xyz'), np.c_[test1[0, :], test1[1, :], test1[2, :]],
                   header='x y z', fmt='%1.6f',
                   delimiter=' ')
        np.savetxt(os.path.join('test', category_id, model_id, str(step) + '_AE.xyz'), np.c_[test2[0, :], test2[1, :], test2[2, :]],
                   header='x y z', fmt='%1.6f',
                   delimiter=' ')
        np.savetxt(os.path.join('test', category_id, model_id, str(step) + '_agent.xyz'), np.c_[test3[0, :], test3[1, :], test3[2, :]],
                   header='x y z', fmt='%1.6f',
                   delimiter=' ')


        if is_first:
            done = False
            np.savetxt(os.path.join('test', category_id, model_id, 'input.xyz'),
                       np.c_[test1[0, :], test1[1, :], test1[2, :]],
                       header='x y z', fmt='%1.6f',
                       delimiter=' ')
        elif reward_D < self.prev_reward_D:
            done = True
            np.savetxt(os.path.join('test', category_id, model_id, 'output.xyz'),
                       np.c_[test1[0, :], test1[1, :], test1[2, :]],
                       header='x y z', fmt='%1.6f',
                       delimiter=' ')

            # for ground truth
            ply_data = PlyData.read(os.path.join('data/shape_net_core_uniform_samples_2048_split/test', category_id, model_id + '.ply'))
            points = ply_data['vertex']
            gt_points = np.vstack([points['x'], points['y'], points['z']])
            np.savetxt(os.path.join('test', category_id, model_id, 'ground_truth.xyz'),
                       np.c_[gt_points[0, :], gt_points[1, :], gt_points[2, :]],
                       header='x y z', fmt='%1.6f',
                       delimiter=' ')

            # visualizing
            input = np.loadtxt(os.path.join('test', category_id, model_id, 'input.xyz'))
            output = test1.T
            gt = gt_points.T
            plot_pcd_three_views(os.path.join('test', category_id, model_id, 'result.png'),
                                 [input, output, gt],
                                 ['input', 'output', 'gt'])


            # cd loss
            chamfer_distance = self.chamfer(torch.Tensor(test1).cuda().unsqueeze(0), torch.Tensor(gt_points).cuda().unsqueeze(0))
            self.cd_loss += chamfer_distance.detach().cpu().numpy()
        else:
            done = False

        self.prev_reward_D = reward_D

        new_state = out_G.detach().cpu().data.numpy().squeeze()
        new_state_pc = torch.transpose(pc_1_G, 1, 2).unsqueeze(1)
        return new_state, new_state_pc, reward, done, self.lossess.avg



if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss

    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())

    print(args)

    """-------------------------------------------------Visualer Initialization-------------------------------------"""

    visualizer = Visualizer(args)

    args.display_id = args.display_id + 10
    args.name = 'Validation'
    vis_Valid = Visualizer(args)
    vis_Valida = []
    args.display_id = args.display_id + 10

    for i in range(1, 30):
        vis_Valida.append(Visualizer(args))
        args.display_id = args.display_id + 10


    main(args,vis_Valid,vis_Valida)







