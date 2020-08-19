## A demo compatible with [CyCADA](https://arxiv.org/pdf/1711.03213.pdf).

This demo training script is borrow from CyCADA. I tried to clean the original CyCADA code up a little bit. To train, please configure your environments following the [official repo](https://github.com/jhoffman/cycada_release) of CyCADA. Then copy both files into the 'scripts' folder, and run:<br>
`bash ./scripts/train_fcn_adda_cpn.sh`

The key modification is following lines. 

```
_, fake_label_t = torch.max(score_t, 1)
_, fake_label_t = torch.max(cpn(im_t, one_hot(fake_label_t,19)),1)
loss_supervised_t = supervised_loss(score_t, fake_label_t, weights=weights)
```
CPN creates pseudo-labels that are more compatible with the scenes, and the labels are then used to supervise training. You may play with your own training script accordingly.
