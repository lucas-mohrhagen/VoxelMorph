import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm
import time
import os
from os.path import join
import wandb

from implicitmorph.utils import surface_plot
from implicitmorph.dataio import PerturbedSamplingScheduler, MIN_NORM, MAX_NORM


from implicitmorph.metrics import write_iou, write_all_basic, write_distribution_plot


def train(opt, rank, model, train_dataloader, val_dataloader, train_dataset, val_dataset, sampler, loss_fn, clip_grad=True):
    summaries_dir = join(opt.root_path, 'summaries')
    checkpoints_dir = join(opt.root_path, 'checkpoints')

    device = f'cuda:{rank}'
    model.cuda()
    model.train()

    wandb.init(project=opt.experiment_name)

    optim = torch.optim.Adam(lr=float(opt.lr), params=model.parameters())
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, verbose=True)
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, pin_memory=False, sampler=sampler, num_workers=0)

    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)


    max_norm = MAX_NORM if opt.curriculum_learning else MIN_NORM
    sampling_scheduler = PerturbedSamplingScheduler(train_dataset, opt.num_epochs, MIN_NORM, max_norm, verbose=False)
    if opt.curriculum_learning:
        print('[INFO] curriculum learning enabled.')
    else:
        print('[INFO] curriculum learning disabled.')


    total_steps = 0
    with tqdm(initial=0, total=len(train_dataloader) * opt.num_epochs) as pbar:
        print(f'[INFO] train for {len(train_dataloader)*opt.num_epochs} iterations.')
        train_losses = []

        for epoch in range(opt.num_epochs):
            sampler.set_epoch(epoch)
            
            losses_per_epoch = []

            for model_input in train_dataloader:
                start_time = time.time()
                optim.zero_grad()

                model_input = {key: value.to(device) for key, value in model_input.items()}

                model_output = model(model_input)
                losses = loss_fn(model_output, model_input, epoch)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    wandb.log({loss_name: loss})
                    train_loss += loss

                train_losses.append(train_loss.item())
                losses_per_epoch.append(train_loss.item())

                scaler.scale(train_loss).backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                scaler.step(optim)
                scaler.update()
                optim.step()

                pbar.update(1)
                total_steps += 1

                if not total_steps % opt.steps_til_summary and rank==0:
                    tqdm.write("Epoch %d, Iteration %d, Total loss %0.6f, iteration time %0.6f" % (epoch, total_steps, train_loss, time.time() - start_time))

            
            # EARLY STOPPING
            epoch_loss = sum(losses_per_epoch) / len(losses_per_epoch)
            if opt.should_early_stop:
                stop, patience_hit = ea.earlystop(epoch_loss)
            
                if stop:
                    print("\033[91m[INFO] early stop at epoch %d\033[0m" % epoch)
                    break
            
            scheduler.step(train_loss)
            sampling_scheduler.step_and_update_norm()

            if not epoch % opt.epochs_til_ckpt and epoch and rank==0:
                checkpoint = {"model": model.state_dict(),
                                "optimizer": optim.state_dict()
                                }
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % total_steps))

                surface_plot(opt, model, train_dataset, total_steps, mode='train')
                write_iou(opt, model_input, model_output, total_steps, mode='train')
                write_all_basic(opt, model_input, model_output, total_steps, mode='train')
                write_distribution_plot(opt, model_input, model_output, total_steps)

                eval(opt, val_dataset, val_dataloader, model, loss_fn, epoch, total_steps, device)

            # save final checkpoint
            if rank == 0:
                checkpoint = {"model": model.state_dict(),
                            "optimizer": optim.state_dict()
                            }
                torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_final.pth'))
        wandb.finish()
        print('done', rank)


def eval(opt, val_dataset, val_dataloader, model, loss_fn, epoch, total_steps, device):
    model.eval()
    for model_input in val_dataloader:
        with torch.no_grad():
            model_input = {key: value.to(device) for key, value in model_input.items()}

            model_output = model(model_input)
            # model_output = {key: value.to(device) for key, value in model_output.items()}
            losses = loss_fn(model_output, model_input, epoch, mode='eval')

            val_loss = 0.
            for loss_name, loss in losses.items():
                wandb.log({f'{loss_name} eval': loss})
                val_loss += loss

            write_iou(opt, model_input, model_output, total_steps, mode='eval')
            write_all_basic(opt, model_input, model_output, total_steps, mode='eval')
            surface_plot(opt, model, val_dataset, total_steps, mode='eval')
    model.train()