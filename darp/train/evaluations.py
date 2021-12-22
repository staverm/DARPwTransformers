from icecream import ic
import torch
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os


def offline_evaluation(
        dataloader, 
        model, 
        sacred, 
        emb_typ, 
        typ, 
        device, 
        best_eval_metric, 
        current_epoch, 
        path_name,
        checkpoint_type,
        criterion,
        full_test=True, 
        saving=True, ):
    """
        Use a dataloader as a validation set to verify the models ability to find the supervision strategy.
        As there is no online interaction with out ennvironement. The nomber of data metrics is minimal.
    """
    max_test_accuracy = 0
    running_loss = 0
    total = 0
    correct = nearest_accuracy = pointing_accuracy = 0
    mean_time_distance, mean_pick_distance, mean_drop_distance, mean_correct_loaded, mean_correct_available = 0, 0, 0, 0, 0

    y_pred, y_sup = [], []
    if full_test :
        eval_name = 'Offline Test'
    else :
        eval_name = 'Supervised Test stats'


    model.eval()
    for i, data in enumerate(dataloader):
        observation, supervised_action = data

        if emb_typ >= 40 :
            world, targets, drivers, positions, time_constraints, prior_kwlg = observation
        else :
            world, targets, drivers, positions, time_constraints = observation

        info_block = [world, targets, drivers]
        if typ in [17, 18, 19]:
            target_tensor = world
        else :
            target_tensor = world[1].unsqueeze(-1).type(torch.LongTensor).to(device)

        model_action = model(info_block,
                            target_tensor,
                            positions=positions,
                            times=time_constraints)


        # model_action = model_action[:,0]
        supervised_action = supervised_action.to(device)

        loss = criterion(model_action.squeeze(1), supervised_action.squeeze(-1))
        total += supervised_action.size(0)
        correct += np.sum((model_action.squeeze(1).argmax(-1) == supervised_action.squeeze(-1)).cpu().numpy())
        running_loss += loss.item()

        y_pred = y_pred +model_action.squeeze(1).argmax(dim=-1).flatten().tolist()
        y_sup = y_sup + supervised_action.squeeze(-1).tolist()

        # Limit train passage to 20 rounds
        if i == 20:
            break

    eval_acc = 100 * correct/total
    eval_loss = running_loss/total

    cf_matrix = confusion_matrix(y_sup, y_pred)
    #Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
    f1_metric = f1_score(y_sup, y_pred, average='weighted')
    #Calculate metrics globally by counting the total true positives, false negatives and false positives.
    f1_metric1 = f1_score(y_sup, y_pred, average='micro')
    #Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    f1_metric2 = f1_score(y_sup, y_pred, average='macro')

    ic('\t-->' + eval_name + 'Réussite: ', eval_acc, '%')
    ic('\t-->' + eval_name + 'F1 :', f1_metric)
    ic('\t-->' + eval_name + 'Loss:', running_loss/total)

    # Model saving. Condition: Better accuracy and better loss
    if saving and full_test and (eval_acc > best_eval_metric[0] or ( eval_acc == best_eval_metric[0] and eval_loss <= best_eval_metric[1] )):
        best_eval_metric[0] = eval_acc
        best_eval_metric[1] = eval_loss
        if checkpoint_type == 'best':
            model_name = path_name + '/models/best_offline_model.pt'
        else :
            model_name = path_name + '/models/model_offline' + str(current_epoch) + '.pt'
        os.makedirs(path_name + '/models/', exist_ok=True)
        ic('\t New Best Accuracy Model <3')
        ic('\tSaving as:', model_name)
        torch.save(model, model_name)

        dir = path_name + '/example/'
        os.makedirs(dir, exist_ok=True)
        save_name = dir + '/cf_matrix.png'
        conf_img = sn.heatmap(cf_matrix, annot=True)
        plt.savefig(save_name)
        plt.clf()
        sacred.get_logger().report_media('Image', 'Confusion Matrix',
                                            iteration=current_epoch,
                                            local_path=save_name)

    # Statistics on clearml saving
    if sacred :
        sacred.get_logger().report_scalar(title=eval_name,
            series='reussite %', value=eval_acc, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Loss', value=running_loss/total, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='F1 score weighted', value=f1_metric, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='F1 score micro', value=f1_metric1, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='F1 score macro', value=f1_metric2, iteration=current_epoch)

    return best_eval_metric

def dataset_evaluation(
        model, 
        sacred, 
        emb_typ, 
        typ, 
        device, 
        best_eval_metric, 
        current_epoch, 
        path_name,
        checkpoint_type,
        inst_name,
        dataset_env ):
    ic('\t** ON DATASET :', inst_name, '**')
    eval_name = 'Dataset Test'
    done = False
    observation = dataset_env.reset()
    total_reward = 0
    while not done:
        if emb_typ >= 40 :
            world, targets, drivers, positions, time_contraints, prior_kwlg = observation
            time_contraints = [torch.tensor([time_contraints[0]], dtype=torch.float64),
                            [torch.tensor([time], dtype=torch.float64) for time in time_contraints[1]]]
        else :
            world, targets, drivers, positions, time_contraints = observation
            time_contraints = [torch.tensor([time_contraints[0]], dtype=torch.float64),
                            [torch.tensor([time], dtype=torch.float64) for time in time_contraints[1]],
                            [torch.tensor([time], dtype=torch.float64) for time in time_contraints[2]]]

        positions = [torch.tensor([positions[0]], dtype=torch.float64),
                        [torch.tensor([position], dtype=torch.float64) for position in positions[1]],
                        [torch.tensor([position], dtype=torch.float64) for position in positions[2]]]

        w_t = [torch.tensor([winfo],  dtype=torch.float64) for winfo in world]
        t_t = [[torch.tensor([tinfo], dtype=torch.float64 ) for tinfo in target] for target in targets]
        d_t = [[torch.tensor([dinfo],  dtype=torch.float64) for dinfo in driver] for driver in drivers]
        info_block = [w_t, t_t, d_t]

        target_tensor = torch.tensor([world[1]]).unsqueeze(-1).type(torch.LongTensor).to(device)

        model_action = model(info_block,
                                    target_tensor,
                                    positions=positions,
                                    times=time_contraints)

        if typ >25:
            chosen_action = model_action.argmax(-1).cpu().item()
        else :
            chosen_action = model_action[:, 0].argmax(-1).cpu().item()

        observation, reward, done, info = dataset_env.step(chosen_action)
        total_reward += reward

    if info['fit_solution'] and info['GAP'] < best_eval_metric[2] :
        ic('/-- NEW BEST GAP SOLUTION --\\')
        ic('/-- GAP:', info['GAP'])
        best_eval_metric[2] = info['GAP']
        if checkpoint_type == 'best':
            model_name = path_name + '/models/best_GAP_model.pt'
        else :
            model_name = path_name + '/models/GAP_model_' + str(current_epoch) + '.pt'
        ic('\tSaving as:', model_name)
        os.makedirs(path_name + '/models/', exist_ok=True)
        torch.save(model, model_name)

    ic('/- Fit solution:', info['fit_solution'])
    ic('/- with ',info['delivered'], 'deliveries')
    if info['fit_solution'] :
        ic('/- GAP to optimal solution: ', info['GAP'], '(counts only if fit solution)')
    ic('/- Optim Total distance:', dataset_env.best_cost)
    ic('/- Model Total distance:', dataset_env.total_distance)

    if sacred :
        sacred.get_logger().report_scalar(title=eval_name,
            series='Fit solution', value=info['fit_solution'], iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Delivered', value=info['delivered'], iteration=current_epoch)
        if info['fit_solution'] > 0:
            sacred.get_logger().report_scalar(title=eval_name,
                series='Average gap', value=info['GAP'], iteration=current_epoch)
        else :
            sacred.get_logger().report_scalar(title=eval_name,
                series='Average gap', value=300, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Total Reward', value=total_reward, iteration=current_epoch)

    return best_eval_metric

def online_evaluation( 
    model, 
    sacred, 
    emb_typ, 
    typ, 
    best_eval_metric, 
    current_epoch, 
    path_name,
    checkpoint_type,
    eval_env,
    eval_episodes,
    supervision,
    example_format,
    criterion,
    device,
    full_test=True, 
    saving=True):
    """
        Online evaluation of the model according to the supervision method.
        As it is online, we  can maximise the testing informtion about the model.
    """
    correct = total = running_loss = total_reward = 0
    delivered = 0
    gap = 0
    fit_sol = 0
    supervision.env = eval_env
    if full_test :
        eval_name = 'Test stats'
    else :
        eval_name = 'Supervised Test stats'

    model.eval()
    for eval_step in range(eval_episodes):

        # Generate solution and evironement instance.
        done = False
        observation = eval_env.reset()

        if example_format == 'svg':
            to_save = [eval_env.get_svg_representation() if full_test else 0]
        else :
            to_save = [eval_env.get_image_representation() if full_test else 0]
        save_rewards = [0]
        last_time = 0

        while not done:
            if emb_typ >= 40 :
                world, targets, drivers, positions, time_contraints, prior_kwlg = observation
                time_contraints = [torch.tensor([time_contraints[0]], dtype=torch.float64),
                                [torch.tensor([time], dtype=torch.float64) for time in time_contraints[1]]]
            else :
                world, targets, drivers, positions, time_contraints = observation
                time_contraints = [torch.tensor([time_contraints[0]], dtype=torch.float64),
                                [torch.tensor([time], dtype=torch.float64) for time in time_contraints[1]],
                                [torch.tensor([time], dtype=torch.float64) for time in time_contraints[2]]]

            positions = [torch.tensor([positions[0]], dtype=torch.float64),
                            [torch.tensor([position], dtype=torch.float64) for position in positions[1]],
                            [torch.tensor([position], dtype=torch.float64) for position in positions[2]]]

            w_t = [torch.tensor([winfo],  dtype=torch.float64) for winfo in world]
            t_t = [[torch.tensor([tinfo], dtype=torch.float64 ) for tinfo in target] for target in targets]
            d_t = [[torch.tensor([dinfo],  dtype=torch.float64) for dinfo in driver] for driver in drivers]
            info_block = [w_t, t_t, d_t]

            if typ in [17, 18, 19]:
                target_tensor = w_t
            else :
                target_tensor = torch.tensor([world[1]]).unsqueeze(-1).type(torch.LongTensor).to(device)

            model_action = model(info_block,
                                        target_tensor,
                                        positions=positions,
                                        times=time_contraints)

            if supervision :
                supervised_action = supervision.action_choice()
                supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(device)

            if typ >25:
                chosen_action = model_action.argmax(-1).cpu().item()
            else :
                chosen_action = model_action[:, 0].argmax(-1).cpu().item()

            if full_test :
                observation, reward, done, info = eval_env.step(chosen_action)
            elif supervision :
                observation, reward, done, info = eval_env.step(supervised_action)

            # self.eval_env.render()
            if supervision :
                loss = criterion(model_action[:,0], supervised_action)
                running_loss += loss.item()
                correct += (chosen_action == supervised_action).cpu().numpy()[0]
            else :
                correct += 0
                running_loss += 0
                loss = 0

            total_reward += reward
            total += 1

            if eval_env.time_step > last_time and full_test:
                last_time = eval_env.time_step
                if example_format == 'svg':
                    to_save.append(eval_env.get_svg_representation())
                else :
                    to_save.append(eval_env.get_image_representation())
                save_rewards.append(reward)
        # Env is done
            # If not rf supervision
        gap += info['GAP']

        fit_sol += info['fit_solution'] #self.eval_env.is_fit_solution()
        delivered += info['delivered']

    # To spare time, only the last example is saved
    eval_acc = 100 * correct/total
    eval_loss = running_loss/total

    ic('\t-->' + eval_name + 'Réussite: ', eval_acc, '%')
    ic('\t-->' + eval_name + 'Loss:', running_loss/total)
    ic('\t-->' + eval_name + 'Fit solution: ', 100*fit_sol/eval_episodes, '%')
    ic('\t-->' + eval_name + 'Average delivered', delivered/eval_episodes)
    ic('\t-->' + eval_name + 'Step Reward ', total_reward/total)

    # Model saving. Condition: Better accuracy and better loss
    if fit_sol > 0 and gap < best_eval_metric[3] :
        best_eval_metric[3] = gap
        if checkpoint_type == 'best':
            model_name = path_name + '/models/best_online_model.pt'
        else :
            model_name = path_name + '/models/model_online' + str(current_epoch) + '.pt'
        os.makedirs(path_name + '/models/', exist_ok=True)
        ic('\t New Best online GAP Model <3')
        ic('\tSaving as:', model_name)
        torch.save(model, model_name)

        # Saving an example
        # if self.example_format == 'svg':
        #     self.save_svg_example(to_save, save_rewards, 0, time_step=self.current_epoch)
        # else :
        #     self.save_example(to_save, save_rewards, 0, time_step=self.current_epoch)


    # Statistics on clearml saving
    if sacred :
        sacred.get_logger().report_scalar(title=eval_name,
            series='reussite %', value=eval_acc, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Loss', value=running_loss/total, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Fit solution %', value=100*fit_sol/eval_episodes, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Average delivered', value=delivered/eval_episodes, iteration=current_epoch)
        if supervision_function == 'rf' :
            if fit_sol > 0:
                sacred.get_logger().report_scalar(title=eval_name,
                    series='Average gap', value=gap/fit_sol, iteration=current_epoch)
            else :
                sacred.get_logger().report_scalar(title=eval_name,
                    series='Average gap', value=300, iteration=current_epoch)
        else :
            sacred.get_logger().report_scalar(title=eval_name,
                series='Average gap', value=gap/eval_episodes, iteration=current_epoch)
        sacred.get_logger().report_scalar(title=eval_name,
            series='Step Reward', value=total_reward/total, iteration=current_epoch)