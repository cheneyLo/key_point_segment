import visdom
import numpy as np
from subprocess import Popen, PIPE
import sys

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer(object):
    def __init__(self, display_env, open=False, **kwargs):
        self.open = open
        self.display_port = 8097
        self.display_server = "http://localhost"
        self.vis = visdom.Visdom(server=self.display_server, port=self.display_port, env='main')
        if not self.vis.check_connection():
            self.create_visdom_connections()
        self.index = {}
        self.cur_win = {}


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &' % self.display_port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def plot_losses(self, d):
        '''
        self.plot('loss',1.00)
        '''
        if self.open:
            name = list(d.keys())
            name_total = " ".join(name)
            x = self.index.get(name_total, 0)
            val = list(d.values())
            if len(val) == 1:
                y = np.array(val)
            else:
                y = np.array(val).reshape(-1, len(val))
            try:
                self.vis.line(Y=y, X=np.ones(y.shape) * x,
                              win=str(name_total),  # unicode
                              opts=dict(legend=name,
                                        title=name_total),
                              update=None if x == 0 else 'append'
                              )
                self.index[name_total] = x + 1
            except:
                pass

    def show_image(self, imagedict):
        if self.open:
            names =  list(imagedict.keys())
            images = list(imagedict.values())
            nlen = len(names)

            try:
                for i in range(nlen):
                    if isinstance(images[i], list):
                        try:
                            for j in range(len(images[i])):
                                images[i][j] = (images[i][j] - images[i][j].min())/images[i][j].max()
                        except:
                            images[i][j] = (images[i][j] - images[i][j].min())
                        image = np.concatenate(images[i], axis=1)
                    else:
                        image = images[i]
                    image = 255 * (image - image.min()) / (image.max() - image.min())
                    image = image.astype(np.uint8)

                    self.vis.images(image, nrow=1, opts=dict(title=names[i], caption='predict vs true'), win=1)
            except:
                pass

    def show_image3D(self, imagedict, win=1):
        if self.open:
            try:
                for key in imagedict:
                    imagelist = imagedict[key]
                    nlen = len(imagelist)
                    images = []
                    for i in range(nlen):
                        for j in range(imagelist[i].shape[0]):
                            if imagelist[i][j,:,:].max() > 0:
                                imagelist[i][j,:,:] =  255 * imagelist[i][j,:,:] / max(imagelist[i][j,:,:].max(), 1)

                        d, h, w = imagelist[i].shape
                        image = imagelist[i].reshape(h*d, w)
                        image = image.astype(np.uint8)
                        images.append(image)

                    images = np.concatenate(images, axis=1)
                    self.vis.images(images, nrow=1, opts=dict(title=key, caption='predict vs true'), win=win)
            except:
                pass

    def processbar(self, epoch, epochs, indeximage, totalimages, opts=None):
        if self.open:
            win = self.cur_win.get("processbar", None)
            bili = epoch / epochs * 100
            imgbili = indeximage / totalimages * 100

            tbl_str = "<div style=\"width: 100%; background-color: #ddd;\" > " \
                      "<div style=\"text-align: right;margin:5px; padding-right: 20px;height: 40px;line-height: 40px;color: white; width: " + (
                                  '%.2f' % bili) + "%;    background-color: #4CAF50;\">" + str(epoch) + "/" + str(
                epochs) + "</div>" \
                          "<div style=\"text-align: right;margin:5px; padding-right: 20px;height: 40px;line-height: 40px;color: white; width: " + (
                                  '%.2f' % imgbili) + "%; background-color: #2196F3;\">" + (
                                  '%.2f' % imgbili) + "%</div>" \
                                                      "</div>"

            default_opts = {'title': 'processbar'}
            if opts is not None:
                default_opts.update({'title': opts})
            try:
                if win is not None:
                    self.vis.text(tbl_str, win=win, opts=default_opts)
                else:
                    self.cur_win["processbar"] = self.vis.text(tbl_str, opts=default_opts)
            except:
                pass
