import os
from concurrent import futures

import torch

import grpc

# from ...constants import GRPC_MAX_MSG_SIZE
from ...models import Model
from ...utils import create_generate_request, print_rank_n
from .pb import generation_pb2, generation_pb2_grpc

import pdb

class GenerationServer(generation_pb2_grpc.GenerationServiceServicer):
    def __init__(self, model: Model) -> None:
        self.model = model

    def _unpack_proto_query_kwargs(self, query_kwargs):
        query_kwargs = {k: getattr(v, v.WhichOneof("oneof_values")) for k, v in query_kwargs.items()}
        return query_kwargs

    def Generate(self, request, context):
        text = [r for r in request.texts]
        generate_kwargs = self._unpack_proto_query_kwargs(request.generate_kwargs)

        request = create_generate_request(text=text, generate_kwargs=generate_kwargs)

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        self.model.input_device = local_rank

        # Check request number of token: if > 0 -> Generate, if = 0, calculate logprob
#         (Pdb) request
# GenerateRequest(text=['DeepSpeed is a machine learning framework'], min_length=None, do_sample=None, early_stopping=None, temperature=1.0, top_k=None, top_p=1.0, typical_p=None, repetition_penalty=None, bos_token_id=None, pad_token_id=None, eos_token_id=None, length_penalty=None, no_repeat_ngram_size=None, encoder_no_repeat_ngram_size=None, num_return_sequences=None, max_time=None, max_new_tokens=64, decoder_start_token_id=None, diversity_penalty=None, forced_bos_token_id=None, forced_eos_token_id=None, exponential_decay_length_penalty=None, remove_input_from_output=False, method='generate')
        #pdb.set_trace()
        if request.max_new_tokens > 0: #generate
            response = self.model.generate(request)
            if isinstance(response, Exception):
            # if exception occurs, we don't this subprocess to crash
                response = generation_pb2.GenerationResponse(error=str(response))
            else:
                response = generation_pb2.GenerationResponse(
                    texts=response.text, num_generated_tokens=response.num_generated_tokens
                )

            return response
    
        else: # calculate logprob of the input            
            try:
                logprobs = self.model.logprob(request.text)
                response = generation_pb2.GenerationResponse(
                    texts=str(logprobs), num_generated_tokens=0
                )
            except Exception as e:
                # if exception occurs, we don't this subprocess to crash
                response = generation_pb2.GenerationResponse(error=str(e))

            return response        

def serve(inference_pipeline, port):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        # options=[
        #     ("grpc.max_send_message_length", GRPC_MAX_MSG_SIZE),
        #     ("grpc.max_receive_message_length", GRPC_MAX_MSG_SIZE),
        # ],
    )
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServer(inference_pipeline), server)
    server.add_insecure_port(f"[::]:{port}")
    print_rank_n("About to start server")
    server.start()
    print_rank_n("Started")
    server.wait_for_termination()
