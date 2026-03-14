from __future__ import annotations

import torch
import torch.nn as nn

from torchdeq.utils.mem import mem_gc, DEQGradCkpt, filter_input, filter_out, reset_grad


class SimpleLinear(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


class MultiInputModule(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, y):
        return self.linear(x) + y


class MultiOutputModule(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc1(x), self.fc2(x)


class TestFilterInput:
    def test_separates_grad_and_nograd(self):
        a = torch.randn(2, 4, requires_grad=True)
        b = torch.randn(2, 4, requires_grad=False)
        c = torch.randn(2, 4, requires_grad=True)

        forward_args, grad_args, grad_idx = filter_input((a, b, c))
        assert len(forward_args) == 3
        assert len(grad_args) == 2
        assert grad_idx == (0, 2)

    def test_non_tensor_passthrough(self):
        a = torch.randn(2, 4, requires_grad=True)
        b = 42  # non-tensor
        forward_args, grad_args, grad_idx = filter_input((a, b))
        assert len(forward_args) == 2
        assert forward_args[1] == 42
        assert len(grad_args) == 1
        assert grad_idx == (0,)

    def test_detaches_tensors(self):
        a = torch.randn(2, 4, requires_grad=True)
        forward_args, _, _ = filter_input((a,))
        # Should be detached (new tensor) but still require grad
        assert forward_args[0].requires_grad
        assert forward_args[0] is not a


class TestFilterOut:
    def test_filters_non_tensor_grads(self):
        out = (torch.randn(2, 4), torch.randn(2, 4))
        out_grad = (torch.randn(2, 4), None)
        out_tensor, out_grad_tensor = filter_out(out, out_grad)
        assert len(out_tensor) == 1
        assert len(out_grad_tensor) == 1

    def test_keeps_all_tensor_grads(self):
        out = (torch.randn(2, 4), torch.randn(2, 4))
        out_grad = (torch.randn(2, 4), torch.randn(2, 4))
        out_tensor, out_grad_tensor = filter_out(out, out_grad)
        assert len(out_tensor) == 2
        assert len(out_grad_tensor) == 2


class TestResetGrad:
    def test_reorders_to_original_positions(self):
        g0 = torch.randn(2, 4)
        g1 = torch.randn(2, 4)
        in_args = (None, None, None)  # 3 original args
        grad_idx = (0, 2)  # grads at positions 0 and 2
        result = reset_grad((g0, g1), in_args, grad_idx)
        assert result[0] is g0
        assert result[1] is None
        assert result[2] is g1


class TestFetchParams:
    def test_fetches_requires_grad_params(self):
        model = SimpleLinear(4)
        params = DEQGradCkpt.fetch_params(model)
        # Linear has weight and bias
        assert len(params) == 2
        assert all(p.requires_grad for p in params)

    def test_accepts_list_of_modules(self):
        m1 = nn.Linear(4, 4)
        m2 = nn.Linear(4, 4)
        params = DEQGradCkpt.fetch_params([m1, m2])
        assert len(params) == 4  # 2 params each


class TestMemGC:
    def test_forward_matches_direct(self):
        torch.manual_seed(0)
        model = SimpleLinear(4)
        x = torch.randn(2, 4)

        direct_out = model(x)
        gc_out = mem_gc(model, (x,))

        assert torch.allclose(direct_out, gc_out)

    def test_gradients_flow_to_params(self):
        torch.manual_seed(0)
        model = SimpleLinear(4)
        x = torch.randn(2, 4, requires_grad=True)

        out = mem_gc(model, (x,))
        out.sum().backward()

        assert model.linear.weight.grad is not None
        assert model.linear.weight.grad.abs().sum() > 0
        assert model.linear.bias.grad is not None

    def test_gradients_flow_to_input(self):
        model = SimpleLinear(4)
        x = torch.randn(2, 4, requires_grad=True)

        out = mem_gc(model, (x,))
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_correctness(self):
        """Gradients from mem_gc should match standard autograd."""
        torch.manual_seed(42)
        model = SimpleLinear(4)
        x = torch.randn(2, 4, requires_grad=True)

        # Standard forward/backward
        out_std = model(x)
        out_std.sum().backward()
        w_grad_std = model.linear.weight.grad.clone()
        x_grad_std = x.grad.clone()

        # Reset grads
        model.zero_grad()
        x.grad = None

        # mem_gc forward/backward
        out_gc = mem_gc(model, (x,))
        out_gc.sum().backward()
        w_grad_gc = model.linear.weight.grad
        x_grad_gc = x.grad

        assert torch.allclose(w_grad_std, w_grad_gc, atol=1e-6)
        assert torch.allclose(x_grad_std, x_grad_gc, atol=1e-6)

    def test_multi_input(self):
        model = MultiInputModule(4)
        x = torch.randn(2, 4, requires_grad=True)
        y = torch.randn(2, 4, requires_grad=True)

        out = mem_gc(model, (x, y))
        out.sum().backward()

        assert model.linear.weight.grad is not None
        assert x.grad is not None
        assert y.grad is not None

    def test_no_in_args(self):
        """mem_gc with no in_args should use empty tuple."""
        class NoArgModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(4))

            def forward(self):
                return self.param.sum()

        model = NoArgModule()
        out = mem_gc(model)
        out.backward()
        assert model.param.grad is not None

    def test_non_grad_input_gets_no_grad(self):
        model = SimpleLinear(4)
        x = torch.randn(2, 4, requires_grad=False)

        out = mem_gc(model, (x,))
        out.sum().backward()

        # Input didn't require grad, so it shouldn't have one
        assert x.grad is None
        # But model params should still get grads
        assert model.linear.weight.grad is not None

    def test_iterative_usage(self):
        """Simulate DEQ-style iterative usage: call mem_gc multiple times in a loop."""
        torch.manual_seed(0)
        model = SimpleLinear(4)
        with torch.no_grad():
            model.linear.weight.mul_(0.1)

        z = torch.zeros(2, 4, requires_grad=True)
        for _ in range(5):
            z = mem_gc(model, (z,))

        z.sum().backward()
        assert model.linear.weight.grad is not None
        assert model.linear.weight.grad.abs().sum() > 0
