#include <llama.h>
#include <stdio.h>

int main(int argc, char** argv){
  struct llama_model_params model_params = llama_model_default_params();
  
  char* model_path = "gemma-3n-E4B-it-UD-IQ2_XXS.gguf";
  
  char* prompt = "hello";

  struct llama_model *model  = llama_model_load_from_file(model_path, model_params);
  struct llama_context_params ctx_params = llama_context_default_params();
  
  struct llama_context *ctx = llama_init_from_model(model, ctx_params);
  
  if (llama_supports_gpu_offload()){
    printf("\nYour GPU is supported\n");
  }
  

  llama_free(ctx);
  llama_model_free(model); 
  return 0;
}
