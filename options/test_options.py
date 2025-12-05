from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
        # Fast mode arguments: allow running a quick subset of directories before full evaluation
        parser.add_argument('--fast', action='store_true', help='Run a quick sanity check on limited number of subdirectories.')
        parser.add_argument('--max_dirs', type=int, default=2, help='Maximum number of subdirectories to process in fast mode.')
        # (Potential future) limit images; currently unused but reserved
        parser.add_argument('--max_images', type=int, default=0, help='If >0 and fast mode enabled, cap number of images per subdirectory.')

        self.isTrain = False
        return parser
