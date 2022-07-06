from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self.is_train = False
