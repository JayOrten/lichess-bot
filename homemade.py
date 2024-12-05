"""
Some example classes for people who want to create a homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""
import chess
from chess.engine import PlayResult, Limit
import random
from lib.engine_wrapper import MinimalEngine
from lib.types import MOVE, HOMEMADE_ARGS_TYPE
import logging
import os
import torch
import yaml
from sentencepiece import SentencePieceProcessor
from pytorch_lightning import LightningModule
from transformers import (
    LlamaForCausalLM as LanguageModel, 
    LlamaConfig as HFConfig
)
from typing import List

# Use this logger variable to print messages to the console or log files.
# logger.info("message") will always print "message" to the console or log file.
# logger.debug("message") will only print "message" if verbose logging is enabled.
logger = logging.getLogger(__name__)

CONFIG_PATH = 'C:\\Users\\jayor\\Documents\\repos\\lichess-bot\\data\\model_config.yaml'

device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')
class cGPT(MinimalEngine):
    """A simple engine that uses the cGPT model to generate moves."""

    def __init__(self, commands, options, stderr: int | None, draw_or_resign, name: str | None = None, **popen_args: str) -> None:
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)
        """
        Initialize the engine.

        :param model_path: The path to the cGPT model.
        """
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        # Convert args dict to object
        self.config = Struct(**config)

        self.tokenizer = Tokenizer(self.config.tokenizer_path)
        self.config.vocab_size = self.tokenizer.n_words
        self.config.pad_id = self.tokenizer.pad_id
        self.model = self.load_model()

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """
        Choose a move using the cGPT model.

        :param board: The current position.
        :return: The move to play.
        """
        return PlayResult(self.make_move(board), None)
    
    def load_model(self):
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(tokenizer=self.tokenizer, config=self.config)

        # Load checkpoint
        checkpoint_path=self.config.checkpoint_path

        print(f"Using checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['state_dict'])

        model = model.model

        model.cuda()
        model.eval()

        return model
    
    def make_move(self, board):
        current_moves = [x for x in board.move_stack]

        if len(current_moves) == 0:
            return board.parse_san('e4')
        
        move_sequence_san = []
        new_board = chess.Board()
        for move in current_moves:
            san = new_board.san(move)
            move_sequence_san.append(san)
            new_board.push_san(san)

        # First, check if greedy move is legal
        k = 1
        while True:
            move = self.gpt_move(move_sequence_san, do_sample=True, top_k=k)
            if not self.is_illegal(board, move):
                break
            k += 1
            if k > 700:
                move = board.san(random.choice(list(board.legal_moves)))
                break

        return board.parse_san(move)
    
    def gpt_move(self, current_sequence, do_sample=True, top_k=1):
        new_sequence = list(current_sequence)
        current_sequence_length = len(current_sequence)
        string_sequence= ' '.join(new_sequence)

        prompt_tokens = torch.tensor(self.tokenizer.encode(string_sequence, bos=True, eos=False)).reshape(1,-1)
        pad_id = self.tokenizer.pad_id
        eos_id = self.tokenizer.eos_id
        bos_id = self.tokenizer.bos_id

        # Return top 10 beams
        generate_ids = self.model.generate(input_ids=prompt_tokens.to(device), 
                                    max_new_tokens=6, 
                                    # num_beams=10,
                                    do_sample=do_sample,
                                    top_k=top_k,
                                    pad_token_id=pad_id,
                                    eos_token_id=eos_id,
                                    bos_token_id=bos_id)
        
        generate_tokens = generate_ids.tolist()

        decoded = self.tokenizer.decode(generate_tokens)

        # If it predicts padding as the next token, just get the last token in the list
        if len(decoded[0].split(' ')) <= current_sequence_length:
            return decoded[0].split(' ')[-1]
        else:
            return decoded[0].split(' ')[current_sequence_length]
    
    def is_illegal(self, board, move):
        try:
            if board.parse_san(move) in board.legal_moves:
                return False
            else:
                return True
        except:
            return True

class Struct:
    """
    Struct class used to convert a dictionary to an object

    Used to serialize the YAML configuration file into a class object,
    where the keys of the dictionary are converted to attributes of the object.
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        s = "Struct: \n"
        for key, value in self.__dict__.items():
            s += f"{key}: {value} \n"
        return s

class Tokenizer:
    """
    Tokenizer class for SentencePiece tokenization
    """
    def __init__(self, model_path):
        assert os.path.exists(model_path), model_path

        self.sp_model = SentencePieceProcessor(model_file=model_path)
        
        logger.info(f"Reloaded SentencePiece model from {model_path}")
    
        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        # NOTE: pad_id is disabled by default with sentencepiece, and the trained llama tokenzier does not use padding
        # If you would like to have a padding token, you can either A) train you own sentencepiece tokenizer
        # or B) add a padding token to the tokenizer, via the 'add_tokens.py' script. This is more janky though.
        self.pad_id: int = self.sp_model.pad_id() # To use modified pad, replace .pad_id() with: ['<pad>'] 

        logger.info(
            f"# of words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        # print(f"\n Pad Token ID: {self.pad_id}\n",f"BOS Token ID: {self.bos_id}\n", f"EOS Token ID: {self.eos_id}\n")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
    
class Model(LightningModule):
    def __init__(self,
                 tokenizer, 
                 config: dict = None):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Load model here
        if config.from_pretrained is not True:
            # * Configure necessary HF model parameters here
            model_config = HFConfig(
                vocab_size = config.vocab_size,
                max_position_embeddings = config.max_sequence_embeddings,
                hidden_size=config.dim,
                num_hidden_layers=config.n_layers,
                num_attention_heads=config.n_heads,
                rms_norm_eps=config.norm_eps,
                pad_token_id=config.pad_id
            )
            self.model = LanguageModel(model_config)
        elif config.from_pretrained is True and config.model_name is not None:
            self.model = LanguageModel.from_pretrained(config.model_name)
        else:
            raise ValueError("Must provide model_name if from_pretrained is True")
        
        self.validation_step_outputs = [] # Used for saving predictions throughout training

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, self.config.gamma)
        return [optimizer], [lr_scheduler]
    
    def monitor_gpu_memory(self):
        """
        Monitor GPU memory usage. Useful for debugging, checking GPU utilization.
        """
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        
class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Bot names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    """
    Get a move using multiple different methods.

    This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
    """

    def search(self, board: chess.Board, time_limit: Limit, ponder: bool, draw_offered: bool, root_moves: MOVE) -> PlayResult:
        """
        Choose a move using multiple different methods.

        :param board: The current position.
        :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
        :param ponder: Whether the engine can ponder after playing a move.
        :param draw_offered: Whether the bot was offered a draw.
        :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
        :return: The move to play.
        """
        if isinstance(time_limit.time, int):
            my_time = time_limit.time
            my_inc = 0
        elif board.turn == chess.WHITE:
            my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
            my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
        else:
            my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
            my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

        if my_time / 60 + my_inc > 10:
            # Choose a random move.
            move = random.choice(possible_moves)
        else:
            # Choose the first move alphabetically in uci representation.
            possible_moves.sort(key=str)
            move = possible_moves[0]
        return PlayResult(move, None, draw_offered=draw_offered)