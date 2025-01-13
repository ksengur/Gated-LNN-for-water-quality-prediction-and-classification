classdef customLiquidLayer < nnet.layer.Layer
    properties (Learnable)
        WeightsInput % Giriş ağırlıkları
        WeightsHidden % Gizli durum ağırlıkları
        Bias % Bias vektörü
    end

    properties
        HiddenState % Dinamik gizli durum
        GatingState % Kapı durumu
    end

    methods
        function layer = customLiquidLayer(numNeurons, varargin)
            % Yapıcı fonksiyon
            if nargin > 1
                layer.Name = varargin{1};
            else
                layer.Name = "CustomLiquidLayer";
            end

            layer.Description = "Gated Liquid Neural Layer";

            % Öğrenilebilir parametrelerin başlangıcı
            layer.WeightsInput = randn(numNeurons, numNeurons) * 0.01;
            layer.WeightsHidden = randn(numNeurons, numNeurons) * 0.01;
            layer.Bias = zeros(numNeurons, 1);
            layer.HiddenState = zeros(numNeurons, 1);
            layer.GatingState = zeros(numNeurons, 1); % Kapı durumu başlangıcı
        end

        function Z = predict(layer, X)
            % Kapı dinamiklerini uygulama
            gate = sigmoid(layer.WeightsInput * X + layer.Bias); % Kapı durumu
            layer.GatingState = gate .* layer.HiddenState; % Kapı etkisi
            
            % Sıvı sinir dinamiği
            layer.HiddenState = tanh(...
                layer.WeightsInput * X + ...
                layer.WeightsHidden * layer.GatingState + ...
                layer.Bias);

            Z = layer.HiddenState;
        end
    end
end

% classdef customLiquidLayer < nnet.layer.Layer
%     properties (Learnable)
%         WeightsInput % Weight matrix for input
%         WeightsHidden % Weight matrix for hidden state
%         Bias % Bias vector
%     end
% 
%     properties
%         HiddenState % Hidden state (dynamic)
%     end
% 
%     methods
%         function layer = customLiquidLayer(numNeurons, varargin)
%             % Constructor
%             if nargin > 1
%                 layer.Name = varargin{1}; % Optional name parameter
%             else
%                 layer.Name = "CustomLiquidLayer";
%             end
% 
%             layer.Description = "Custom Liquid Neural Layer with Gated Dynamics";
% 
%             % Initialize learnable parameters
%             layer.WeightsInput = randn(numNeurons, numNeurons) * 0.01;
%             layer.WeightsHidden = randn(numNeurons, numNeurons) * 0.01;
%             layer.Bias = zeros(numNeurons, 1);
%             layer.HiddenState = zeros(numNeurons, 1); % Initial hidden state
%         end
% 
%         function Z = predict(layer, X)
%             % Liquid Neural dynamics with hidden state
%             layer.HiddenState = tanh(...
%                 layer.WeightsInput * X + ...
%                 layer.WeightsHidden * layer.HiddenState + ...
%                 layer.Bias);
% 
%             Z = layer.HiddenState;
%         end
%     end
% end
% 
% % classdef customLiquidLayer < nnet.layer.Layer
% %     properties (Learnable)
% %         Weights % Weight matrix
% %         Bias    % Bias vector
% %     end
% % 
% %     methods
% %         function layer = customLiquidLayer(numNeurons, varargin)
% %             % Constructor
% %             if nargin > 1
% %                 layer.Name = varargin{1}; % Optional name parameter
% %             else
% %                 layer.Name = "CustomLiquidLayer";
% %             end
% % 
% %             layer.Description = "Custom Liquid Neural Layer";
% % 
% %             % Initialize learnable parameters
% %             layer.Weights = randn(numNeurons, numNeurons) * 0.01;
% %             layer.Bias = zeros(numNeurons,1);
% %         end
% % 
% %         function Z = predict(layer, X)
% %             % Liquid Neural dynamics
% %             Z = tanh(layer.Weights*X + layer.Bias);
% %         end
% %     end
% % end
